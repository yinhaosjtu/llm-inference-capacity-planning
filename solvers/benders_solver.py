# Callback-based multi-cut Benders decomposition solver with energy-gradient-augmented cuts.
import os
import math
import time
import threading
import gurobipy as gp
from gurobipy import GRB
from concurrent.futures import ThreadPoolExecutor

from solvers import gurobi_solver


# Multi-cut Benders solver combining lazy-constraint callbacks and outer gap certification.
class BendersSolver:

    def __init__(self, inst, time_limit=1200, gap_tol=1e-4, lambda_override=None,
                 n_threads=None,
                 scale_threshold=1500,
                 enable_mip_start=None,
                 n_sub_workers=None,
                 parallel_min_scenarios=64):
        self.inst = inst
        self.time_limit = time_limit
        self.gap_tol = gap_tol
        self.n_threads = n_threads
        self.lam = inst['lambda_risk'] if lambda_override is None else lambda_override
        self._extract()
        self._precompute()

        self.nS = len(self.S)
        large = self.nS >= scale_threshold
        self.use_mip_start = large if enable_mip_start is None else enable_mip_start

        cpu = os.cpu_count() or 1
        req = cpu if n_sub_workers is None else n_sub_workers
        if self.nS < parallel_min_scenarios:
            req = 1
        self.n_sub_workers = max(1, min(req, self.nS))

        self._envs = []
        self.master = None
        self._sub = {}
        self._owner = {}
        self._worker_scen = []
        self._executor = None
        self._cuts = 0
        self._sub_failed = False
        self._interrupted = False
        self._cb_lock = threading.Lock()
        self._max_outer = 50
        self._cut_tol_abs = 1e-6
        self._cut_tol_rel = 1e-7
        self._sub_opt_tol = 1e-9
        self._conv_rel = gap_tol
        self._conv_abs = 1e-6
        self._last_eval = None

    def _extract(self):
        I = self.inst
        self.K, self.J, self.S = I['K'], I['J'], I['S']
        self.Psi = set(map(tuple, I['Psi_incomp']))
        self.C_cap, self.C_elast, self.C_rej = I['C_cap'], I['C_elast'], I['C_rej']
        self.C_energy, self.M, self.N_bar = I['C_energy'], I['M'], I['N_bar']
        self.mu_kv, self.xi, self.beta = I['mu_kv'], I['xi'], I['beta']
        self.sigma_out = I['sigma_out']
        self.sigma_tot = I.get('sigma_tot', self.sigma_out)
        self.Z_min, self.Z_max = I['Z_min'], I['Z_max']
        self.Theta, self.T_slot, self.T_op = I['Theta'], I['T_slot'], I['T_op']
        self.P_idle, self.P_peak = I['P_idle'], I['P_peak']
        self.rho, self.kappa = I['rho'], I['kappa']
        self.W_bar, self.E_bar, self.alpha = I['W_bar'], I['E_bar'], I['alpha']
        self.L_in = I['L_in']
        probs = {s: float(I['p'][s]) for s in self.S}
        total_p = math.fsum(probs.values())
        if total_p <= 0:
            raise ValueError("Scenario probabilities must have positive total mass")
        self.p = {s: v / total_p for s, v in probs.items()}
        self.D, self.L_out, self.nu = I['D'], I['L_out'], I['nu_scen']
        self.gpu_tiers = I.get('gpu_tiers', {})
        self._scen_in = bool(I.get('random_input', False)) or (
            len(self.L_in) > 0 and isinstance(next(iter(self.L_in)), tuple))

    def _Lin(self, j, s):
        return self.L_in[(j, s)] if self._scen_in else self.L_in[j]

    # Precompute memory, safety-buffer, and incremental energy coefficients.
    def _precompute(self):
        self.comp = [(j, k) for j in self.J for k in self.K if (j, k) not in self.Psi]
        self.energy_coef_x, self.mem_coef = {}, {}
        for s in self.S:
            for (j, k) in self.comp:
                self.energy_coef_x[(j, k, s)] = (
                    self.C_energy * self.rho * self.T_op *
                    (self.P_peak[k] - self.P_idle[k]) / self.M[k] *
                    (self._Lin(j, s) + self.L_out[(j, s)]) * self.mu_kv * self.xi)
                self.mem_coef[(j, k, s)] = (
                    (self._Lin(j, s) + self.L_out[(j, s)]) * self.mu_kv * self.xi)
        self.vol_coef = {j: self.beta * self.sigma_tot[j] * self.mu_kv * self.xi
                         for j in self.J}
        self.energy_coef_n = {k: -self.C_energy * self.rho * self.T_op *
                              (self.P_peak[k] - self.P_idle[k]) / 2 for k in self.K}
        self._theta_floor = sum(self.energy_coef_n[k] * self.N_bar[k] for k in self.K)
        self._dscale = {s: max(1.0, max(self.D[(j, s)] for j in self.J)) for s in self.S}

    def _tol(self, q):
        return self._cut_tol_abs + self._cut_tol_rel * abs(q)

    def _make_env(self):
        e = gp.Env(empty=True)
        try:
            e.setParam('OutputFlag', 0)
            e.setParam('Threads', 1)
            e.start()
            return e
        except Exception:
            try:
                e.dispose()
            except Exception:
                pass
            raise

    # Build the LP recourse subproblem for a single scenario s.
    def _build_sub_model(self, env, s):
        sub = gp.Model(env=env)
        sub.Params.OutputFlag = 0
        sub.Params.Method = 1
        sub.Params.NumericFocus = 3
        sub.Params.OptimalityTol = self._sub_opt_tol
        sub.Params.FeasibilityTol = self._sub_opt_tol
        sub.Params.InfUnbdInfo = 1
        x = sub.addVars(self.comp, lb=0, name="x")
        e = sub.addVars(self.comp, lb=0, name="e")
        r = sub.addVars(self.J, lb=0, name="r")
        sc = self._dscale[s]
        obj = gp.LinExpr()
        for (j, k) in self.comp:
            obj += self.C_elast[k] * sc * e[j, k] + self.energy_coef_x[(j, k, s)] * x[j, k]
        for j in self.J:
            obj += self.C_rej[j] * sc * r[j]
        sub.setObjective(obj, GRB.MINIMIZE)

        for j in self.J:
            lhs = gp.LinExpr()
            for k in self.K:
                if (j, k) not in self.Psi:
                    lhs += (self.nu[(j, k, s)] / sc) * x[j, k] + e[j, k]
            lhs += r[j]
            sub.addConstr(lhs == self.D[(j, s)] / sc)

        mem, vol, tput = {}, {}, {}
        for k in self.K:
            lm = gp.LinExpr()
            for j in self.J:
                if (j, k) not in self.Psi:
                    lm += self.mem_coef[(j, k, s)] * x[j, k]
            mem[k] = sub.addConstr(lm <= 0.0)
        for k in self.K:
            lv = gp.LinExpr()
            for j in self.J:
                if (j, k) not in self.Psi:
                    lv += self.vol_coef[j] * x[j, k]
            vol[k] = sub.addConstr(lv <= 0.0)
        for k in self.K:
            lt = gp.LinExpr()
            for j in self.J:
                if (j, k) not in self.Psi:
                    lt += self.L_out[(j, s)] * x[j, k]
            tput[k] = sub.addConstr(lt <= 0.0)
        sub.update()
        return sub, mem, vol, tput

    # Partition scenario subproblems across isolated single-thread Gurobi environments.
    def _setup_subproblems(self):
        W = self.n_sub_workers
        self._envs = []
        try:
            for _ in range(W):
                self._envs.append(self._make_env())
        except gp.GurobiError:
            for env in self._envs:
                try:
                    env.dispose()
                except Exception:
                    pass
            self._envs = []
            W = self.n_sub_workers = 1
            self._envs = [self._make_env()]
        self._worker_scen = [[] for _ in range(W)]
        for idx, s in enumerate(self.S):
            w = idx % W
            self._owner[s] = w
            self._worker_scen[w].append(s)
        for w in range(W):
            env = self._envs[w]
            for s in self._worker_scen[w]:
                self._sub[s] = self._build_sub_model(env, s)
        if W > 1:
            self._executor = ThreadPoolExecutor(max_workers=W)

    # Solve a subproblem LP with method fallbacks (dual simplex -> barrier -> primal simplex).
    def _solve_sub(self, sub):
        sub.optimize()
        if sub.Status == GRB.OPTIMAL:
            return True
        if sub.Status == GRB.INTERRUPTED:
            self._interrupted = True
            return False
        try:
            sub.Params.NumericFocus = 3
            sub.reset()
            sub.optimize()
            if sub.Status == GRB.OPTIMAL:
                return True
            if sub.Status == GRB.INTERRUPTED:
                self._interrupted = True
                return False
            sub.Params.Method = 2
            sub.reset()
            sub.optimize()
            if sub.Status == GRB.OPTIMAL:
                return True
            if sub.Status == GRB.INTERRUPTED:
                self._interrupted = True
                return False
            sub.Params.Method = 0
            sub.reset()
            sub.optimize()
            if sub.Status == GRB.INTERRUPTED:
                self._interrupted = True
                return False
            return sub.Status == GRB.OPTIMAL
        finally:
            sub.Params.Method = 1

    def _solve_chunk(self, scen, n_val, y_val):
        out = {}
        for s in scen:
            sub, mem, vol, tput = self._sub[s]
            for k in self.K:
                mem[k].RHS = (n_val[k] - y_val[k]) * self.M[k]
                vol[k].RHS = y_val[k] * self.M[k]
                tput[k].RHS = n_val[k] * self.Theta[k] * self.T_slot
            if self._solve_sub(sub):
                duals = {'mu': {k: mem[k].Pi for k in self.K},
                         'zeta': {k: vol[k].Pi for k in self.K},
                         'gamma': {k: tput[k].Pi for k in self.K}}
                out[s] = (sub.ObjVal, duals)
            else:
                out[s] = None
        return out

    # Solve scenario subproblems at (n_val, y_val) across worker threads.
    def _solve_all_subs(self, n_val, y_val, scenarios=None):
        scen = self.S if scenarios is None else scenarios
        if self.n_sub_workers <= 1 or len(scen) < 2 or self._executor is None:
            return self._solve_chunk(scen, n_val, y_val)
        groups = {}
        for s in scen:
            groups.setdefault(self._owner[s], []).append(s)
        results = {}
        futs = [self._executor.submit(self._solve_chunk, g, n_val, y_val)
                for g in groups.values()]
        for f in futs:
            results.update(f.result())
        return results

    def _theta_lb(self):
        return {s: self._theta_floor for s in self.S}

    # Build the Benders master MILP with McCormick envelopes and CVaR constraints.
    def _build_master(self, theta_lb):
        mp = gp.Model("Benders_Master")
        mp.Params.OutputFlag = 0
        mp.Params.LazyConstraints = 1
        mp.Params.MIPGap = self.gap_tol
        mp.Params.MIPGapAbs = 1e-6
        mp.Params.FeasibilityTol = 1e-9
        mp.Params.OptimalityTol = 1e-9
        mp.Params.IntFeasTol = 1e-9
        mp.Params.NumericFocus = 3
        if self.n_threads is not None:
            mp.Params.Threads = self.n_threads

        n = mp.addVars(self.K, vtype=GRB.INTEGER, lb=0,
                       ub={k: self.N_bar[k] for k in self.K}, name="n")
        z = mp.addVars(self.K, vtype=GRB.CONTINUOUS, lb=self.Z_min, ub=self.Z_max, name="z")
        y = mp.addVars(self.K, lb=0, name="y")
        eta = mp.addVar(lb=-GRB.INFINITY, name="eta")
        phi = mp.addVars(self.S, lb=0, name="phi")
        theta = mp.addVars(self.S, lb=theta_lb, name="theta")

        mp.addConstr(self.rho * gp.quicksum(self.P_peak[k] * n[k] for k in self.K) <= self.W_bar)
        E_base = self.rho * self.T_op * gp.quicksum(
            (self.P_idle[k] + self.P_peak[k]) / 2 * n[k] for k in self.K)
        mp.addConstr(self.kappa * E_base <= self.E_bar)
        for k in self.K:
            Nk = self.N_bar[k]
            mp.addConstr(y[k] >= self.Z_min * n[k])
            mp.addConstr(y[k] >= self.Z_max * n[k] + z[k] * Nk - self.Z_max * Nk)
            mp.addConstr(y[k] <= self.Z_max * n[k])
            mp.addConstr(y[k] <= self.Z_min * n[k] + z[k] * Nk - self.Z_min * Nk)
        for s in self.S:
            mp.addConstr(phi[s] >= theta[s] - eta)

        mp.setObjective(
            gp.quicksum(self.C_cap[k] * n[k] for k in self.K) + self.C_energy * E_base
            + (1 - self.lam) * gp.quicksum(self.p[s] * theta[s] for s in self.S)
            + self.lam * (eta + (1 / (1 - self.alpha)) *
                          gp.quicksum(self.p[s] * phi[s] for s in self.S)),
            GRB.MINIMIZE)

        self.master, self.n, self.z, self.y = mp, n, z, y
        self.eta, self.phi, self.theta = eta, phi, theta

    def _require_all(self, res):
        for s in self.S:
            if res.get(s) is None:
                raise RuntimeError(
                    "Benders subproblem could not be solved to optimality during initialization")

    # Project candidate (n, y) onto integer supply and McCormick bounds.
    def _clean_point(self, n_raw, y_raw):
        n_val, y_val = {}, {}
        for k in self.K:
            nk = min(self.N_bar[k], max(0, round(n_raw[k])))
            lo = self.Z_min * nk
            hi = self.Z_max * nk
            yk = y_raw[k]
            if yk < lo:
                yk = lo
            elif yk > hi:
                yk = hi
            n_val[k] = float(nk)
            y_val[k] = yk
        return n_val, y_val

    def _same_point(self, a_n, a_y, b_n, b_y):
        if a_n is None or a_y is None or b_n is None or b_y is None:
            return False
        for k in self.K:
            if abs(a_n[k] - b_n[k]) > 1e-6:
                return False
            if abs(a_y[k] - b_y[k]) > max(1e-6, 1e-8 * max(1.0, abs(a_y[k]), abs(b_y[k]))):
                return False
        return True

    def _store_eval(self, n_val, y_val, Q, res):
        self._last_eval = (dict(n_val), dict(y_val), Q, res)
        return n_val, y_val, Q, res

    # Evaluate exact augmented recourse costs Q(n, y, s) for all scenarios.
    def _eval_first_stage(self, n_raw, y_raw, use_cache=True):
        n_val, y_val = self._clean_point(n_raw, y_raw)
        if use_cache and self._last_eval is not None:
            cached_n, cached_y, Q, res = self._last_eval
            if self._same_point(n_val, y_val, cached_n, cached_y):
                return cached_n, cached_y, Q, res
        res = self._solve_all_subs(n_val, y_val)
        for s in self.S:
            if res.get(s) is None:
                return None
        ec = sum(self.energy_coef_n[k] * n_val[k] for k in self.K)
        Q = {s: res[s][0] + ec for s in self.S}
        return self._store_eval(n_val, y_val, Q, res)

    def _cvar(self, Q):
        tail = 1.0 - self.alpha
        if tail <= 1e-12:
            return max(Q[s] for s in self.S)
        order = sorted(((Q[s], self.p[s]) for s in self.S),
                       key=lambda t: t[0], reverse=True)
        acc, tot = 0.0, 0.0
        for c, pr in order:
            if acc + pr < tail:
                tot += c * pr
                acc += pr
            else:
                tot += c * (tail - acc)
                acc = tail
                break
        return tot / tail if acc > 0 else order[0][0]

    def _objective_value(self, n_val, Q):
        capital = sum(self.C_cap[k] * n_val[k] for k in self.K)
        E_base = self.rho * self.T_op * sum(
            (self.P_idle[k] + self.P_peak[k]) / 2 * n_val[k] for k in self.K)
        base_energy = self.C_energy * E_base
        exp_rec = sum(self.p[s] * Q[s] for s in self.S)
        cvar = self._cvar(Q)
        return capital + base_energy + (1 - self.lam) * exp_rec + self.lam * cvar

    def _add_cuts_at(self, n_raw, y_raw):
        ev = self._eval_first_stage(n_raw, y_raw)
        if ev is None:
            if self._interrupted:
                return None
            self._require_all({})
        n_val, y_val, Q, res = ev
        for s in self.S:
            const, cut = self._gen_cut(s, Q[s], res[s][1], n_val, y_val)
            self.master.addConstr(self.theta[s] >= const + cut)
        return Q

    def _set_risk_starts(self, Q):
        for s in self.S:
            self.theta[s].Start = Q[s]
        order = sorted(self.S, key=lambda s: Q[s])
        if not order:
            return
        acc = 0.0
        eta_star = Q[order[-1]]
        for s in order:
            acc += self.p[s]
            if acc >= self.alpha:
                eta_star = Q[s]
                break
        self.eta.Start = eta_star
        for s in self.S:
            self.phi[s].Start = max(0.0, Q[s] - eta_star)

    # Seed initial Benders cuts (with a baseline-mean warm start on large instances).
    def _seed_master(self):
        if self.use_mip_start:
            warm = None
            try:
                ev = gurobi_solver._build_mean_instance(self.inst)
                warm = gurobi_solver.solve(ev, time_limit=self.time_limit, gap_tol=self.gap_tol,
                                           lambda_override=self.lam, return_first_stage=True,
                                           scenario_input=self._scen_in)
            except KeyboardInterrupt:
                self._interrupted = True
                return
            except Exception:
                warm = None
            if warm is not None and warm.get('status') == GRB.INTERRUPTED:
                self._interrupted = True
                return
            if warm is not None and warm.get('n') is not None:
                n_val = {k: float(warm['n'][k]) for k in self.K}
                warm_y = warm.get('y')
                y_val = {k: (float(warm_y[k]) if warm_y is not None
                             else float(warm['n'][k]) * float(warm['z'][k]))
                         for k in self.K}
                Q = self._add_cuts_at(n_val, y_val)
                if self._interrupted:
                    return
                for k in self.K:
                    self.n[k].Start = warm['n'][k]
                    self.z[k].Start = warm['z'][k]
                    self.y[k].Start = y_val[k]
                self._set_risk_starts(Q)
                return
        n_val = {k: float(self.N_bar[k]) for k in self.K}
        y_val = {k: float(self.Z_min) * float(self.N_bar[k]) for k in self.K}
        self._add_cuts_at(n_val, y_val)

    # Construct energy-gradient-augmented optimality cut from subproblem duals.
    def _gen_cut(self, s, Q_s, duals, n_val, y_val):
        mu, zeta, gamma = duals['mu'], duals['zeta'], duals['gamma']
        cut = gp.LinExpr()
        const = Q_s
        for k in self.K:
            coef_n = (mu[k] * self.M[k] + gamma[k] * self.Theta[k] * self.T_slot
                      + self.energy_coef_n[k])
            coef_y = -mu[k] * self.M[k] + zeta[k] * self.M[k]
            cut += coef_n * self.n[k] + coef_y * self.y[k]
            const -= coef_n * n_val[k] + coef_y * y_val[k]
        return const, cut

    def solve(self):
        t0 = time.time()
        self._cuts = 0
        deadline = t0 + self.time_limit

        # Lazy-constraint callback invoked at integer-feasible master solutions.
        def cb(model, where):
            if where != GRB.Callback.MIPSOL:
                return
            with self._cb_lock:
                if self._sub_failed or self._interrupted:
                    return
                n_raw = {k: model.cbGetSolution(self.n[k]) for k in self.K}
                y_raw = {k: model.cbGetSolution(self.y[k]) for k in self.K}
                n_val, y_val = self._clean_point(n_raw, y_raw)
                th = {s: model.cbGetSolution(self.theta[s]) for s in self.S}
                res = self._solve_all_subs(n_val, y_val)
                for s in self.S:
                    if res.get(s) is None:
                        if not self._interrupted:
                            self._sub_failed = True
                        model.terminate()
                        return
                ec = sum(self.energy_coef_n[k] * n_val[k] for k in self.K)
                Q = {s: res[s][0] + ec for s in self.S}
                self._store_eval(n_val, y_val, Q, res)
                for s in self.S:
                    Q_s = Q[s]
                    if Q_s > th[s] + self._tol(Q_s):
                        const, cut = self._gen_cut(s, Q_s, res[s][1], n_val, y_val)
                        model.cbLazy(self.theta[s] >= const + cut)
                        self._cuts += 1

        best_ub = float('inf')
        LB = -float('inf')
        status = GRB.TIME_LIMIT
        have_inc = False
        inc_n, inc_y = None, None
        try:
            self._interrupted = False
            self._setup_subproblems()
            self._build_master(self._theta_lb())
            self._seed_master()
            if self._interrupted:
                status = GRB.INTERRUPTED
            else:
                # Outer certification loop verifying UB - LB optimality gap.
                for _ in range(self._max_outer):
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        status = GRB.TIME_LIMIT
                        break
                    self.master.Params.TimeLimit = max(0.0, remaining)
                    self._sub_failed = False
                    self.master.optimize(cb)
                    if self._interrupted or self.master.Status == GRB.INTERRUPTED:
                        status = GRB.INTERRUPTED
                        break
                    if self._sub_failed:
                        status = GRB.NUMERIC
                        break
                    if self.master.SolCount == 0:
                        status = self.master.Status
                        break
                    try:
                        bnd = self.master.ObjBound
                    except Exception:
                        bnd = -float('inf')
                    if bnd > LB:
                        LB = bnd
                    n_raw = {k: self.n[k].X for k in self.K}
                    y_raw = {k: self.y[k].X for k in self.K}
                    ev = self._eval_first_stage(n_raw, y_raw)
                    if ev is None:
                        status = GRB.INTERRUPTED if self._interrupted else GRB.NUMERIC
                        break
                    n_val, y_val, Q, res = ev
                    ub = self._objective_value(n_val, Q)
                    if ub < best_ub:
                        best_ub = ub
                        inc_n, inc_y = dict(n_val), dict(y_val)
                    have_inc = True
                    denom = max(1.0, abs(best_ub))
                    if (best_ub - LB) <= max(self._conv_abs, self._conv_rel * denom):
                        status = GRB.OPTIMAL
                        break
                    if self.master.Status != GRB.OPTIMAL:
                        status = self.master.Status
                        break
                    added = 0
                    for s in self.S:
                        th_s = self.theta[s].X
                        if Q[s] > th_s + self._tol(Q[s]):
                            const, cut = self._gen_cut(s, Q[s], res[s][1], n_val, y_val)
                            self.master.addConstr(self.theta[s] >= const + cut)
                            self._cuts += 1
                            added += 1
                    if added == 0:
                        for s in self.S:
                            th_s = self.theta[s].X
                            if Q[s] > th_s + 1e-9 * max(1.0, abs(Q[s])):
                                const, cut = self._gen_cut(s, Q[s], res[s][1], n_val, y_val)
                                self.master.addConstr(self.theta[s] >= const + cut)
                                self._cuts += 1
                                added += 1
                    if added == 0:
                        status = GRB.SUBOPTIMAL
                        break
                else:
                    status = GRB.TIME_LIMIT
        except (gp.GurobiError, MemoryError) as ex:
            msg = str(ex).lower()
            is_oom = isinstance(ex, MemoryError) or 'memory' in msg \
                or getattr(ex, 'errno', None) == 10001
            status = GRB.MEM_LIMIT if is_oom else GRB.NUMERIC
        except RuntimeError:
            status = GRB.NUMERIC
        finally:
            self._cleanup()

        out = {'status': status, 'obj_val': None, 'gap': None,
               'time': time.time() - t0, 'cuts': self._cuts}
        if have_inc and best_ub < float('inf'):
            out['obj_val'] = best_ub
            denom = max(1.0, abs(best_ub))
            out['gap'] = max(0.0, (best_ub - LB) / denom)
            if inc_n is not None:
                z_inc = {}
                for k in self.K:
                    nk = inc_n[k]
                    z_inc[k] = min(self.Z_max, max(self.Z_min, inc_y[k] / nk)) if nk >= 1 else self.Z_min
                out['_inc'] = {
                    'n': {k: int(round(inc_n[k])) for k in self.K},
                    'z': {k: float(z_inc[k]) for k in self.K},
                    'y': {k: float(inc_y[k]) for k in self.K},
                }
        return out

    # Dispose worker pool, master/subproblem models, and Gurobi environments.
    def _cleanup(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        try:
            self.master.dispose()
        except Exception:
            pass
        for s in list(self._sub):
            try:
                self._sub[s][0].dispose()
            except Exception:
                pass
        self._sub.clear()
        for env in self._envs:
            try:
                env.dispose()
            except Exception:
                pass
        self._envs = []


# Entry point; optionally extracts detailed recourse metrics via a fixed-stage EF LP solve.
def solve(inst, time_limit=1200, gap_tol=1e-4, lambda_override=None, n_threads=None,
          return_first_stage=False, return_details=False, **kwargs):
    solver = BendersSolver(inst, time_limit=time_limit, gap_tol=gap_tol,
                           lambda_override=lambda_override, n_threads=n_threads,
                           **kwargs)
    res = solver.solve()
    inc = res.pop('_inc', None)

    if not (return_first_stage or return_details):
        return res

    if inc is None or res.get('status') == GRB.INTERRUPTED:
        if return_first_stage:
            res['n'] = res['z'] = res['y'] = None
        return res

    if return_first_stage:
        res['n'] = inc['n']
        res['z'] = inc['z']
        res['y'] = inc['y']

    if return_details:
        lam = inst['lambda_risk'] if lambda_override is None else lambda_override
        scen_in = bool(inst.get('random_input', False)) or (
            len(inst['L_in']) > 0 and isinstance(next(iter(inst['L_in'])), tuple))
        det = gurobi_solver.solve(
            inst, time_limit=time_limit, gap_tol=gap_tol,
            lambda_override=lam,
            fix_first_stage={'n': inc['n'], 'y': inc['y']},
            return_details=True, scenario_input=scen_in)
        if det.get('status') == GRB.INTERRUPTED:
            res['status'] = GRB.INTERRUPTED
            res['obj_val'] = None
            res['gap'] = None
            return res
        _detail_keys = ('n_vals', 'z_vals', 'capital_cost', 'energy_cost',
                        'expected_recourse', 'scenario_recourse', 'cvar_95',
                        'total_gpus', 'avg_z', 'cloud_burst_ratio', 'tier_counts')
        if 'capital_cost' not in det:
            raise RuntimeError(
                "Benders return_details: EF detail solve did not produce details "
                "(status=%s, obj=%s). Refusing to return partial/empty details."
                % (det.get('status'), det.get('obj_val')))
        if det.get('obj_val') is not None and res.get('obj_val') is not None:
            rel = abs(det['obj_val'] - res['obj_val']) / max(1.0, abs(res['obj_val']))
            if rel > 1e-4:
                raise RuntimeError(
                    "Benders return_details: EF-on-fixed obj %.6f disagrees with "
                    "Benders obj %.6f (rel=%.2e)." % (det['obj_val'], res['obj_val'], rel))
        for key in _detail_keys:
            if key in det:
                res[key] = det[key]
        active = [k for k in inc['n'] if inc['n'][k] > 0]
        if active:
            res['avg_z'] = sum(inc['z'][k] for k in active) / len(active)
        else:
            res['avg_z'] = 0.0
        res['z_vals'] = {k: round(inc['z'][k], 4) for k in inc['n']}
    return res

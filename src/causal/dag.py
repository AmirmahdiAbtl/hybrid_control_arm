import itertools


def dag_parents(edges):
    """node -> set of parents, for every node of the edge list."""
    par = {}
    for e in edges:
        par.setdefault(e[0], set())
        par.setdefault(e[1], set()).add(e[0])
    return par


def _children(par):
    """node -> set of children."""
    ch = {v: set() for v in par}
    for v, ps in par.items():
        for p in ps:
            ch.setdefault(p, set()).add(v)
    return ch


def ancestors(par, nodes):
    """All ancestors of ``nodes`` (excluding the nodes themselves unless they are ancestors of each other)."""
    out, todo = set(), list(nodes)
    while todo:
        for p in par.get(todo.pop(), ()):
            if p not in out:
                out.add(p)
                todo.append(p)
    return out


def descendants(par, nodes):
    """All descendants of ``nodes``."""
    ch, out, todo = _children(par), set(), list(nodes)
    while todo:
        for c in ch.get(todo.pop(), ()):
            if c not in out:
                out.add(c)
                todo.append(c)
    return out


def find_cycle(par):
    """A directed cycle as a node list (first node repeated at the end), or [] if the graph is acyclic."""
    ch = _children(par)
    state = dict.fromkeys(ch, 0)  # 0 = unvisited, 1 = on the current path, 2 = done

    def visit(v, path):
        state[v] = 1
        path.append(v)
        for c in sorted(ch[v]):
            if state[c] == 1:
                return path[path.index(c) :] + [c]
            if state[c] == 0:
                cyc = visit(c, path)
                if cyc:
                    return cyc
        state[v] = 2
        path.pop()
        return []

    for v in sorted(ch):
        if state[v] == 0:
            cyc = visit(v, [])
            if cyc:
                return cyc
    return []


def d_separated(edges, xs, ys, zs):
    """d-separation test on the moralized ancestral graph (Lauritzen).

    The ancestral graph of xs | ys | zs is moralized (co-parents joined, directions dropped) and zs removed;
    xs and ys are d-separated iff no node of ys is reachable from xs.

    Parameters
    ----------
    edges : list of tuple
        Directed edges (parent, child, ...).
    xs, ys, zs : iterable of str
        The two node sets to test and the conditioning set.

    Returns
    -------
    bool
        True iff xs and ys are d-separated given zs (False if xs and ys share a node).
    """
    par = dag_parents(edges)
    xs, ys, zs = set(xs), set(ys), set(zs)
    if xs & ys:
        return False
    keep = xs | ys | zs
    keep |= ancestors(par, keep)
    moral = {v: set() for v in keep}
    for v in keep:
        parents = [p for p in par.get(v, ()) if p in keep]
        for p in parents:
            moral[v].add(p)
            moral[p].add(v)
        for p, q in itertools.combinations(parents, 2):
            moral[p].add(q)
            moral[q].add(p)
    seen, todo = set(xs), list(xs)
    while todo:
        v = todo.pop()
        for w in moral[v]:
            if w in zs or w in seen:
                continue
            if w in ys:
                return False
            seen.add(w)
            todo.append(w)
    return True


def trial_graph(edges):
    """The within-trial graph: every edge touching "S" or "SEL" removed."""
    return [e for e in edges if e[0] not in ("S", "SEL") and e[1] not in ("S", "SEL")]


def work_graph(edges):
    """G_work: (parent, child) pairs of the edges that carry no assumption id."""
    return [(e[0], e[1]) for e in edges if len(e) < 4 or e[3] is None]


def is_backdoor_set(edges, t, y, z):
    """Pearl's back-door criterion for adjustment set ``z`` and the effect of ``t`` on ``y``.

    Parameters
    ----------
    edges : list of tuple
        Directed edges (parent, child, ...).
    t, y : str
        Treatment and outcome nodes.
    z : iterable of str
        Candidate adjustment set.

    Returns
    -------
    bool
        True iff no element of z descends from t and t is d-separated from y given z after removing t's outgoing
        edges.
    """
    if set(z) & descendants(dag_parents(edges), {t}):
        return False
    return d_separated([e for e in edges if e[0] != t], {t}, {y}, set(z))


def is_s_admissible(edges, z, s="S", y="Y", t="horTh", sel="SEL"):
    """S-admissibility of ``z`` for transporting Y(0) across sources (Pearl & Bareinboim selection diagram).

    Parameters
    ----------
    edges : list of tuple
        Directed edges (parent, child, ...) including the source node ``s`` and the selection node ``sel``.
    z : iterable of str
        Candidate cross-source adjustment set.
    s, y, t, sel : str
        Source, outcome, treatment and selection nodes.

    Returns
    -------
    bool
        True iff, with the edges into ``t`` removed, ``y`` is d-separated from ``s`` given z, t and sel.
    """
    return d_separated([e for e in edges if e[1] != t], {s}, {y}, set(z) | {t, sel})


def minimal_sets(test, pool):
    """Minimal subsets of ``pool`` that pass ``test``.

    Subsets are enumerated by size and supersets of sets already found are skipped, so every returned set is
    minimal. The search is exhaustive, which is fine for the 7-node pools of gate G2.

    Parameters
    ----------
    test : callable
        set of str -> bool (e.g. a back-door or S-admissibility check).
    pool : list of str
        Candidate nodes.

    Returns
    -------
    list of list of str
        The minimal passing subsets, in enumeration order.
    """
    found = []
    for k in range(len(pool) + 1):
        for c in itertools.combinations(pool, k):
            if not any(set(f) <= set(c) for f in found) and test(set(c)):
                found.append(list(c))
    return found


def dag_report(edges, assumptions, z_t, z_s, used_trial, used_source):
    """Identification checks of gate G2 on the full and the working graph.

    Parameters
    ----------
    edges : list of tuple
        (parent, child, kind, assumption id or None, basis) for G_full.
    assumptions : dict
        Assumption id -> description (every removed edge's id must be present).
    z_t, z_s : list of str
        Within-trial and cross-source adjustment sets.
    used_trial, used_source : iterable of str
        DAG nodes the within-trial and cross-source models actually use (e.g. ``TRIAL_COLUMNS.keys()``).

    Returns
    -------
    dict
        n_edges_full, n_edges_work, acyclic_full, acyclic_work, cycle_full, cycle_work, zt_valid_work,
        zt_valid_full, zs_adm_work, zs_adm_full, zs_plus_grade_adm (must be False), zs_valid_trial (W3; False
        expected), minimal_trial_sets, minimal_s_sets, threats (one dict per removed edge: assumption, edge,
        breaks_zt, breaks_zs), features_ok_trial, features_ok_source, censoring_ok_work, censoring_ok_full,
        probes_ok.
    """
    full = [(e[0], e[1]) for e in edges]
    work = work_graph(edges)
    z_t, z_s = list(z_t), list(z_s)
    pool = list(dict.fromkeys(z_t + z_s))

    def zt_ok(g, z):
        return is_backdoor_set(trial_graph(g), "horTh", "Y", z)

    # Each removed edge is put back on its own to see which adjustment set its assumption protects.
    threats = []
    for e in edges:
        if len(e) > 3 and e[3] is not None:
            g = work + [(e[0], e[1])]
            threats.append(
                {
                    "assumption": e[3],
                    "edge": f"{e[0]} -> {e[1]}",
                    "breaks_zt": not zt_ok(g, z_t),
                    "breaks_zs": not is_s_admissible(g, z_s),
                }
            )
    cyc_full, cyc_work = find_cycle(dag_parents(full)), find_cycle(dag_parents(work))
    return {
        "n_edges_full": len(full),
        "n_edges_work": len(work),
        "acyclic_full": not cyc_full,
        "acyclic_work": not cyc_work,
        "cycle_full": cyc_full,
        "cycle_work": cyc_work,
        "zt_valid_work": zt_ok(work, z_t),
        "zt_valid_full": zt_ok(full, z_t),
        "zs_adm_work": is_s_admissible(work, z_s),
        "zs_adm_full": is_s_admissible(full, z_s),
        "zs_plus_grade_adm": is_s_admissible(work, z_s + ["tgrade"]),
        "zs_valid_trial": zt_ok(work, z_s),
        "minimal_trial_sets": minimal_sets(lambda z: zt_ok(work, z), pool),
        "minimal_s_sets": minimal_sets(lambda z: is_s_admissible(work, z), pool),
        "threats": threats,
        "features_ok_trial": set(used_trial) == set(z_t),
        "features_ok_source": set(used_source) == set(z_s),
        "censoring_ok_work": d_separated(work, {"C"}, {"Y"}, set(z_t) | {"horTh", "S"}),
        "censoring_ok_full": d_separated(full, {"C"}, {"Y"}, set(z_t) | {"horTh", "S"}),
        "probes_ok": all(t["assumption"] in assumptions for t in threats),
    }


def dag_dot(edges, latent):
    """Graphviz DOT source of the DAG (for ``st.graphviz_chart``).

    Measured nodes are boxes (horTh and Y bold), latent nodes dashed ellipses, S / SEL diamonds; edges removed in
    G_work are dashed red with their assumption id as label.

    Parameters
    ----------
    edges : list of tuple
        (parent, child, kind, assumption id or None, ...) for G_full.
    latent : iterable of str
        Unmeasured nodes.

    Returns
    -------
    str
        DOT source with every node of ``edges``.
    """
    latent = set(latent)
    nodes = list(dict.fromkeys([e[0] for e in edges] + [e[1] for e in edges]))
    lines = ["digraph DAG {", "  rankdir=LR;", '  node [fontname="Helvetica", fontsize=11];']
    for v in nodes:
        if v in ("S", "SEL"):
            style = "shape=diamond"
        elif v in latent:
            style = "shape=ellipse, style=dashed"
        else:
            style = "shape=box, style=bold" if v in ("horTh", "Y") else "shape=box"
        lines.append(f'  "{v}" [{style}];')
    for e in edges:
        assumption = e[3] if len(e) > 3 else None
        attrs = f' [style=dashed, color=red, fontcolor=red, label="{assumption}"]' if assumption else ""
        lines.append(f'  "{e[0]}" -> "{e[1]}"{attrs};')
    lines.append("}")
    return "\n".join(lines)

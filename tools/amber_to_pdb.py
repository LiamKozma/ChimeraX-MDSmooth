#!/usr/bin/env python3
"""Write a topology PDB from an Amber prmtop + first frame of a NetCDF trajectory.

ChimeraX has no Amber prmtop reader, so to load an Amber trajectory you need a
structure file that defines the atoms. This writes a PDB whose atom order is
identical to the trajectory, so the .nc coordinates load cleanly onto it:

    python amber_to_pdb.py system.prmtop production.nc topology.pdb

Then in ChimeraX:

    open topology.pdb
    open production.nc format amber structureModel #1
    mdsmooth #1 toAtoms #1@CA

Requires numpy + scipy (both ship with ChimeraX; also pip-installable).
"""
import sys


def read_flag(text, flag):
    """Return the raw data lines for a %FLAG section of an Amber prmtop."""
    i = text.index("%FLAG " + flag)
    lines = text[i:].split("\n")
    out = []
    for ln in lines[2:]:  # skip the %FLAG and %FORMAT lines
        if ln.startswith("%FLAG") or ln.startswith("%COMMENT"):
            break
        out.append(ln)
    return out


def parse_a4(lines, n):
    """Parse fixed-width 4-char fields (Amber FORMAT 20a4)."""
    vals = []
    for ln in lines:
        for k in range(0, len(ln), 4):
            field = ln[k:k + 4]
            if len(field) == 4:
                vals.append(field.strip())
        if len(vals) >= n:
            break
    return vals[:n]


def parse_ints(lines, n):
    vals = []
    for ln in lines:
        vals.extend(int(x) for x in ln.split())
        if len(vals) >= n:
            break
    return vals[:n]


def element_of(name):
    for ch in name:
        if ch.isalpha():
            return ch
    return "C"


# Residue-name classes used to split the single Amber "chain" into real chains.
# Amber prmtops carry no chain IDs or TER records, so without this every molecule
# is written as one continuous chain and ChimeraX draws spurious bonds across the
# gaps (e.g. a long bond joining two separate protein chains, or protein->ligand).
_AMINO = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # protonation / cap variants Amber uses
    "HID", "HIE", "HIP", "CYX", "CYM", "ASH", "GLH", "LYN", "ACE", "NME", "NHE",
}
_SOLVENT = {
    "WAT", "HOH", "TIP3", "TIP", "SPC", "T3P",
    "NA", "NA+", "CL", "CL-", "K", "K+", "MG", "MG2", "CA", "ZN", "IP", "IM",
}
_CHAIN_IDS = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
              "abcdefghijklmnopqrstuvwxyz0123456789")


def _residue_class(rlab):
    if rlab in _AMINO:
        return "protein"
    if rlab in _SOLVENT:
        return "solvent"
    return "ligand"


def assign_chains(res_labels, res_atom_names, res_atom_coords):
    """Return (chain_id, is_hetatm) per residue.

    A new chain starts when the residue class changes, or when two consecutive
    protein residues are not peptide-bonded (backbone C(i)->N(i+1) > 2.0 A --
    the tell-tale of a chain break in a merged Amber system). All solvent shares
    one chain (waters are never peptide-linked, so they need no breaks) and a
    covalently-continuous ligand (e.g. an oligosaccharide) stays one chain.
    """
    nres = len(res_labels)
    chain_of = [""] * nres
    het_of = [False] * nres
    ci = 0
    prev_cls = None
    prev_c = None  # backbone C coord of previous protein residue
    for r in range(nres):
        cls = _residue_class(res_labels[r])
        names = res_atom_names[r]
        coords = res_atom_coords[r]
        n_coord = coords[names.index("N")] if "N" in names else None
        if r > 0:
            new_chain = cls != prev_cls
            if not new_chain and cls == "protein":
                if prev_c is not None and n_coord is not None:
                    d = sum((a - b) ** 2 for a, b in zip(prev_c, n_coord)) ** 0.5
                    new_chain = d > 2.0
                else:
                    new_chain = True
            if new_chain:
                ci += 1
        chain_of[r] = _CHAIN_IDS[ci % len(_CHAIN_IDS)]
        het_of[r] = cls != "protein"
        prev_cls = cls
        prev_c = coords[names.index("C")] if "C" in names else None
    return chain_of, het_of


def main(prmtop, ncfile, outpdb):
    from scipy.io import netcdf_file

    with open(prmtop) as f:
        text = f.read()

    natom, nres = parse_ints(read_flag(text, "POINTERS"), 12)[0:12:11]
    atom_names = parse_a4(read_flag(text, "ATOM_NAME"), natom)
    res_labels = parse_a4(read_flag(text, "RESIDUE_LABEL"), nres)
    res_ptr = parse_ints(read_flag(text, "RESIDUE_POINTER"), nres)  # 1-based

    atom_res = [0] * natom
    for r in range(nres):
        start = res_ptr[r] - 1
        end = res_ptr[r + 1] - 1 if r + 1 < nres else natom
        for a in range(start, end):
            atom_res[a] = r

    nc = netcdf_file(ncfile, "r", mmap=False)
    coords = nc.variables["coordinates"][0]
    if coords.shape[0] != natom:
        raise SystemExit("prmtop has %d atoms but trajectory has %d -- mismatched files"
                         % (natom, coords.shape[0]))

    # Group atom names / coords per residue so we can detect real chain breaks.
    res_atom_names = [[] for _ in range(nres)]
    res_atom_coords = [[] for _ in range(nres)]
    for a in range(natom):
        res_atom_names[atom_res[a]].append(atom_names[a])
        res_atom_coords[atom_res[a]].append(tuple(coords[a]))
    chain_of, het_of = assign_chains(res_labels, res_atom_names, res_atom_coords)

    with open(outpdb, "w") as out:
        prev_chain = None
        nchains = 0
        for a in range(natom):
            r = atom_res[a]
            chain = chain_of[r]
            if prev_chain is not None and chain != prev_chain:
                out.write("TER\n")  # break connectivity between chains/molecules
            if chain != prev_chain:
                nchains += 1
            prev_chain = chain
            name = atom_names[a]
            aname = (" " + name) if len(name) < 4 else name
            rlab = res_labels[r][:3]
            rnum = (r % 9999) + 1
            x, y, z = coords[a]
            record = "HETATM" if het_of[r] else "ATOM  "
            out.write(
                "%6s%5d %-4s %-3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n"
                % (record, a + 1, aname, rlab, chain, rnum, x, y, z,
                   element_of(name).rjust(2))
            )
        out.write("TER\nEND\n")
    print("Wrote %s: %d atoms, %d residues, %d chains" % (outpdb, natom, nres, nchains))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3])

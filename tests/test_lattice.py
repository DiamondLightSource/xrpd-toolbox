from xrpd_toolbox.fit_engine.lattice import CubicLattice, Lattice


def test_cubic_lattice():
    cl = CubicLattice(a=5)
    params = cl.get_refinement_parameters()
    params["a"] = 5.5
    cl.set_refinement_parameters(params)

    params = cl.get_refinement_parameters()

    assert params["a"] == 5.5

    model_dict = cl.model_dump()
    cl2 = Lattice(**model_dict)

    assert isinstance(cl2, Lattice)

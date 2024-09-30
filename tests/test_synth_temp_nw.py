import pytest

def test_distro():
    from flowstab.SynthTempNetwork import (
        Distro,
    )
    d = Distro(loc=0.0, scale=1.0)
    _ = d.draw_val()
    _ = d.draw_val(loc=0.5)
    _ = d.draw_val(scale=0.5)
    _ = d.draw_val(loc=0.5, scale=1.5)


# you are here!
@pytest.mark.parametrize(
        "_id, i_d_loc, i_d_scale, i_d_mf, a_d_loc, a_d_scale, a_d_mf, dist_type, group", 
        [(1, 0.0, 1.0, None, 0.0, 1.0, None, "experimental", 0),]
    )
def test_individual(_id,
                    i_d_loc, i_d_scale, i_d_mf,
                    a_d_loc, a_d_scale, a_d_mf,
                    dist_type, group):
    from flowstab.SynthTempNetwork import (
        Individual,
    )
    i1 = Individual(ID=_id,
                    inter_distro_loc=i_d_loc,
                    inter_distro_scale=i_d_scale,
                    inter_distro_type=dist_type,
                    inter_distro_mod_func=i_d_mf,
                    activ_distro_loc=a_d_loc,
                    activ_distro_scale=a_d_scale,
                    activ_distro_type=dist_type,
                    activ_distro_mod_func=a_d_mf,
                    group=group)
    _ = i1.draw_inter_duration(time=None)
    _ = i1.draw_inter_duration(time=1.0)
    _ = i1.draw_activ_time(time=None)
    _ = i1.draw_activ_time(time=1.0)

def test_synth_temp_network():
    from flowstab.SynthTempNetwork import (
        Individual,
        SynthTempNetwork
    )
    individuals = [Individual(i, group=0) for i in range(20)]
    sim = SynthTempNetwork(individuals, t_start=0, t_end=50)

    sim.run(save_all_states=True, save_dt_states=True, verbose=True)


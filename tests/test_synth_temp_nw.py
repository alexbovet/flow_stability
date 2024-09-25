from flowstab.SynthTempNetwork import (
    Individual,
    SynthTempNetwork
)

def test_distro():
    from flowstab.SynthTempNetwork import (
        Distro,
    )
    d = Distro(loc=0.0, scale=1.0)
    _ = d.draw_val()
    _ = d.draw_val(loc=0.5)
    _ = d.draw_val(scale=0.5)
    _ = d.draw_val(loc=0.5, scale=1.5)

def test_individual():
    from flowstab.SynthTempNetwork import (
        Individual,
    )


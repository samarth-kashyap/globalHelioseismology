# {{{ Library imports
import argparse
import globalHelioseismology as GH

# {{{ ArgumentParser
parser = argparse.ArgumentParser()
parser.add_argument("--l1",
                    help="spherical harmonic degree for mode 1",
                    type=int, default=50)
parser.add_argument("--l2",
                    help="spherical harmonic degree for mode 2",
                    type=int, default=52)
parser.add_argument("--n1", help="radial order for mode 1", type=int, default=5)
parser.add_argument("--n2", help="radial order for mode 2", type=int, default=5)
parser.add_argument("--t", help="m - mp", type=int, default=0)
parser.add_argument("--plot",
                    help="plot results after computation",
                    action="store_true", default=True)
ARGS = parser.parse_args()
# }}} parser

cs = GH.spectra.crossSpectra(ARGS.n1, ARGS.l1, ARGS.n2, ARGS.l2, ARGS.t,
                             plot_data=ARGS.plot, plot_snr=True)

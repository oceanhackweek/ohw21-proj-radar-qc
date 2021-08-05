from radarqc.csfile import CSFile

import matplotlib.pyplot as plt
import numpy as np


def main():
    path = "../../codar/CSS_ASSA_21_05_23_0030.cs"
    with open(path, "rb") as f:
        cs = CSFile.load_from(f)
        print(cs.header)

        spectrum = cs.spectrum.to_numpy()
        spectrum = (spectrum - spectrum.min()) / (
            spectrum.max() - spectrum.min()
        )

        spectrum = np.log(1 + spectrum) / np.log(2)
        plt.imshow(spectrum)
        plt.show()


if __name__ == "__main__":
    main()

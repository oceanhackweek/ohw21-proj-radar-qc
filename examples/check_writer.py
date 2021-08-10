import matplotlib.pyplot as plt

from radarqc import csfile


def plot_spectrum(cs: csfile.CSFile):
    plt.imshow(cs.antenna3, aspect=4)
    plt.colorbar()
    plt.show()


def timer(fn, *args, **kwargs):
    import time

    ts = time.time()
    ret = fn(*args, **kwargs)
    return ret, time.time() - ts


def main():
    path = "../../codar/CSS_ASSA_21_06_26_1400.cs"
    with open(path, "rb") as f:
        cs, t = timer(csfile.load, f)
        print(t)

    with open("temp.cs", "wb") as f:
        _, t = timer(csfile.dump, cs, f)
        print(t)


if __name__ == "__main__":
    main()

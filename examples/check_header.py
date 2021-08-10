from radarqc import csfile


def main():
    path = "../../codar/CSS_ASSA_21_06_26_1400.cs"
    with open(path, "rb") as f:
        cs = csfile.load(f)
        print(cs.header)


if __name__ == "__main__":
    main()

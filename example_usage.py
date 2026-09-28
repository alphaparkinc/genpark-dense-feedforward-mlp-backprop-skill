from client import DenseMLP

def main():
    mlp = DenseMLP([3, 5, 2])
    inp = [0.5, -0.2, 0.9]
    out = mlp.forward(inp)
    print("MLP Forward Output 3->5->2:", [round(v, 4) for v in out])

if __name__ == "__main__":
    main()

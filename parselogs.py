import pandas as pd
import os

columns = []

def parse_file(directory, algo, file):
    path = os.path.join(directory, algo, file)
    with open(path, 'r') as f:
        lines = pd.Series(f.readlines()).str[:-1]
        
    maxiter = int(lines[lines.str.contains("maxIter")].values[0].split("=")[1].strip())
    nx = int(lines[lines.str.contains("nx")].values[0].split("=")[1].strip())
    ny = int(lines[lines.str.contains("ny")].values[0].split("=")[1].strip())
    main = float(lines[lines.str.contains("main")].values[0].split("|")[1].strip().split()[1])
    
    equilibrium = float(lines[lines.str.contains("equilibrium")].values[0].split("|")[1].strip().split()[1]) \
                * int(lines[lines.str.contains("equilibrium")].values[0].split("|")[0].split(":")[1].split("=")[1])
    collision = float(lines[lines.str.contains("collision")].values[0].split("|")[1].strip().split()[1]) \
                * int(lines[lines.str.contains("collision")].values[0].split("|")[0].split(":")[1].split("=")[1])
    streaming = float(lines[lines.str.contains("streaming")].values[0].split("|")[1].strip().split()[1]) \
                * int(lines[lines.str.contains("streaming")].values[0].split("|")[0].split(":")[1].split("=")[1])
    macroscopic = float(lines[lines.str.contains("macroscopic")].values[0].split("|")[1].strip().split()[1]) \
                * int(lines[lines.str.contains("macroscopic")].values[0].split("|")[0].split(":")[1].split("=")[1])
    rightwall = float(lines[lines.str.contains("rightwall")].values[0].split("|")[1].strip().split()[1]) \
                * int(lines[lines.str.contains("rightwall")].values[0].split("|")[0].split(":")[1].split("=")[1])
    leftwall = float(lines[lines.str.contains("leftwall")].values[0].split("|")[1].strip().split()[1]) \
                * int(lines[lines.str.contains("leftwall")].values[0].split("|")[0].split(":")[1].split("=")[1])
    fin_inflow = float(lines[lines.str.contains("fin_inflow")].values[0].split("|")[1].strip().split()[1]) \
                * int(lines[lines.str.contains("fin_inflow")].values[0].split("|")[0].split(":")[1].split("=")[1])
    bounceback = float(lines[lines.str.contains("bounceback")].values[0].split("|")[1].strip().split()[1]) \
                * int(lines[lines.str.contains("bounceback")].values[0].split("|")[0].split(":")[1].split("=")[1])
    
    return pd.Series(name=file, 
              data = [algo, maxiter, nx, ny, main, equilibrium, collision, streaming,
                      macroscopic, rightwall, leftwall, fin_inflow, bounceback],
              index = ["algo", "iter", "nx", "ny", "main", "equilibrium", "collision", "streaming",
                      "macroscopic", "rightwall", "leftwall", "fin_inflow", "bounceback"])
    

if __name__ == "__main__":
    directory = "logs"
    algo = "numba"
    files = os.listdir(os.path.join(directory, algo))
    df = pd.DataFrame()
    for f in files:
        row = parse_file(directory, algo, f)
        df = df.append(row, ignore_index=True)
    print(df)
    df.to_csv("perfs.csv", index=False)
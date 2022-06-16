import os

if __name__ == '__main__':
    parsed = []
    for freq in os.listdir('results'):
        for model in os.listdir(f'results/{freq}'):
            with open(f'results/{freq}/{model}') as file:
                [parsed.append(f'{freq.split("f")[1].split("_")[0]},{model.split("_")[0]},{line.replace("-1", "Greedy")}') for line in
                 file.readlines()[1:]]

    parsed.sort(key=lambda p: (int(p.split(',')[0]), p.split(',')[1], p.split(',')[2]))
    parsed = ['Frequency,Model,Batch Size,Average Power(W),Time(s),Energy(J),Average Wait Time(s),Max Wait Time(s),Average Peak Power (W)\n'] + parsed
    parsed = ''.join(f'{p}' for p in parsed)
    print(parsed)

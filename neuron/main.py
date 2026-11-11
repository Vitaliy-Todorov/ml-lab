from model import Neuron


def main():
    neuron = Neuron()
    a = float(input("Введите a: "))
    b = float(input("Введите b: "))
    print(f"Сравнение {a} и {b}: {neuron.target_label(a, b)}")
    print(f"Оценка {a} и {b}: {neuron.predict_probability(a, b)}")
    neuron.train()
    print(f"Оценка {a} и {b}: {neuron.predict_probability(a, b)}")


if __name__ == "__main__":
    main()

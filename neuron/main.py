from model import Neuron

def main():

    while True:
        run_neuron()
        
        text = input("Введите q для выхода: ")

        if text.lower() == "q":
            break


def run_neuron():
    a = float(input("Введите a: "))
    b = float(input("Введите b: "))

    neuron = Neuron()
    print(f"Сравнение {a} и {b}: {neuron.target_label(a, b)}")
    print(f"Оценка {a} и {b}: {neuron.predict_probability(a, b)}")

    training_steps = int(input("\nВведите число шагов обучения: "))
    neuron.train(training_steps)
    print(f"Обновлённая оценка {a} и {b}: {neuron.predict_probability(a, b)}")


if __name__ == "__main__":
    main()

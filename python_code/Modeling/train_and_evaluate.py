import pandas as pd
import tkinter as tk
import seaborn as sns
import matplotlib.pyplot as plt

from buildModel import TrainModel
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from tkinter.filedialog import askopenfilename

root = tk.Tk()
root.withdraw()

def train():
    model_name = input('Enter model file name: ')

    train_model = TrainModel(model_name)
    best_hyperband_param, hyperband_results, best_hyperband_model = train_model.train_model()

    return train_model, best_hyperband_param, hyperband_results, best_hyperband_model

def plot_model_history(history):
    history_df = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    history_df.loc[0:, ['loss', 'val_loss']].plot(ax=ax[0])
    ax[0].set(xlabel='epochs', ylabel='loss')

    history_df.loc[0:, ['accuracy', 'val_accuracy']].plot(ax=ax[1])
    ax[1].set(xlabel='epochs', ylabel='accuracy')

def plot_confusion_matrix(y, y_hat, title='Confusion Matrix'):
  cm = confusion_matrix(y, y_hat)

  sns.heatmap(cm, cmap='PuBu', annot=True, fmt='g', annot_kws={'size': 20})

  plt.xlabel('Predicted', fontsize=10)
  plt.ylabel('Actual', fontsize=10)
  plt.title(title, fontsize=10)

train_model, best_hyperband_param, hyperband_results, best_hyperband_model = train()

model_file = askopenfilename(title="Select model")
model = load_model(model_file)

loss, accuracy = model.evaluate(train_model.test_generator)
print(f'Face Mask Detection using CNN and Hyperband tuner has loss: {loss}, accuracy: {accuracy}')

while True:
    plot_question = input('Do you want to show model plot history ? ')

    if (plot_question == 'yes') or (plot_question == 'YES') or (plot_question == 'Yes'):
        plot_model_history(best_hyperband_model.history)
        plt.show()
        break
    elif (plot_question == 'no') or (plot_question == 'NO') or (plot_question == 'No'):
        break
    else:
        print('The answer must be yes/no')

while True:
    plot_question = input('Do you want to show confusion matrix ? ')

    if (plot_question == 'yes') or (plot_question == 'YES') or (plot_question == 'Yes'):
        y_true = train_model.test_generator.labels
        y_pred = model.predict(train_model.test_generator).argmax(axis=1)

        plot_confusion_matrix(y_true, y_pred)
        plt.show()
        break
    elif (plot_question == 'no') or (plot_question == 'NO') or (plot_question == 'No'):
        break
    else:
        print('The answer must be yes/no')
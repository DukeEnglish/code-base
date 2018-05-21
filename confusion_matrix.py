'''''compute confusion matrix 
labels.txt: contain label name. 
predict.txt: predict_label true_label 
'''  
from sklearn.metrics import confusion_matrix  
import matplotlib.pyplot as plt  
import numpy as np  
import seaborn as sns

# print confused matrix 
cm = confusion_matrix(y_true, y_pred)  
print (cm)  
np.set_printoptions(precision=2)  
cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]  
print( cm_normalized  )

def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True)
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Your code goes here
plt.figure()
plot_confusion_matrix(cm_normalized, classes=['ham', 'spam'])    

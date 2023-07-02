import math
import matplotlib.pyplot as plt
from snakeMLpj.numpy_transformations import mcol, mrow

graph_prop=[6, 4]
fig_max_width=3

def find_dimensions(total, max_width=fig_max_width):
    cols=max_width
    rows=math.ceil(total/cols)
    return rows, cols


def histogram_attributeVSfrequency(data, labels, features, label_names, is_label_dict=False,row_attributes=False, dense=False, save = False, center_data=True, colors=[None, None], folder="", show=False):
    rows, cols=find_dimensions(len(features))
    plt.figure(figsize=(cols*graph_prop[0],rows*graph_prop[1]), dpi=200)
    if center_data:
        if row_attributes:
            data=data-mcol(data.mean(axis=1))
        else:
            data=data-mrow(data.mean(axis=0))
    if is_label_dict:
        lab=list(label_names.keys())
    else:
        lab=label_names
    for i in range(len(features)):
        plt.subplot(rows,cols,i+1)
        plt.xlabel(features[i])
        for j in range(len(lab)):
            if row_attributes:
                plt.hist(data[:, labels==j][i, :], density = dense, label = lab[j], color=colors[0] if j==0 else colors[1])        
            else:
                plt.hist(data[labels==j, :][:, i], density = dense, label = lab[j], color=colors[0] if j==0 else colors[1])        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    if save:
        path=folder+"hist"+ ("_dense" if dense else "_notdense") + ("_centered" if center_data else "_notcentered")+ ".png"
        plt.savefig(path)
        print(path, " saved")
    if show:
        plt.show()

def scatter_attributeVSattribute(data, labels, features, label_names, is_label_dict=False,row_attributes=False, dense=False, save = False, center_data=False, colors=[None, None], folder="", show=False, columns=fig_max_width, name=""):
    rows, cols=find_dimensions((((len(features)-1)*len(features))/2) if len(features)>1 else 1, max_width=columns)
    plt.figure(figsize=(cols*graph_prop[0],rows*graph_prop[1]), dpi=200)
    if center_data:
        if row_attributes:
            data=data-mcol(data.mean(axis=1))
        else:
            data=data-mrow(data.mean(axis=0))
    if is_label_dict:
        lab=list(label_names.keys())
    else:
        lab=label_names
    counter=1
    for i in range(len(features)):
        for k in range(len(features)):
            if i >= k:
                continue
            plt.subplot(rows,cols,counter)
            counter+=1
            plt.xlabel(features[i])
            plt.ylabel(features[k])
            for j in range(len(lab)):
                if row_attributes:
                    plt.scatter(data[:, labels==j][i, :],data[:, labels==j][k, :], label = lab[j], color=colors[0] if j==0 else colors[1])        
                else:
                    plt.scatter(data[labels==j, :][:, i],data[labels==j, :][:, k], label = lab[j], color=colors[0] if j==0 else colors[1])        
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    if show:
        plt.show()
    if save:
        path=folder+"scatter"+name+".png"
        plt.savefig(folder+"scatter"+name+".png")
        print(path, "saved")
    

def scatter_categories(data, labels, label_names, is_label_dict=False,row_attributes=False,  save = False):
    if is_label_dict:
        lab=list(label_names.keys())
    else:
        lab=label_names
    plt.figure()
    for j in range(len(lab)):
        if row_attributes:
            plt.scatter(data[:, labels==j],data[:, labels==j], label = lab[j])        
        else:
            plt.scatter(data[labels==j, :],data[labels==j, :], label = lab[j])        
    plt.legend()
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    if save:
        plt.savefig('scatter_%d_%d.png')
    plt.show()
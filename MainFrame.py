import os
from  tkinter import *
from tkinter.ttk import Combobox
from Processing import *

# edit and build the main screen

window=Tk()
window.title('Data Mining Project')
width =1200
height = 750
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x = (screen_width / 2) - (width / 2)
y = (screen_height / 2) - (height / 2)
window.geometry("%dx%d+%d+%d" % (width, height, x, y-35))
window.resizable(0, 0)

Path = StringVar()
Discretization = StringVar()
NumOFBins = StringVar()
NumOfNeighbors = StringVar()
Algorithm = StringVar()

def InputsForm():
    '''
    function that build a main frame that contain all the entrys (inputs) to give to the user the ability to choose
    what algorithm he want and to fill the other entrys.
    :return:nothing
    '''
    global Inputs_Frame, result_label,NumOFBins_entry,NumOfNeighbors_entry,NumOfNeighbors_label
    Inputs_Frame = Frame(window)
    Inputs_Frame.pack(side=TOP, pady=40)

    Titel_laebl = Label(Inputs_Frame, text="Please fill all the required fields", font=('arial', 18),fg="blue")
    Titel_laebl.grid(row=0, columnspan=2)

    Path_label = Label(Inputs_Frame, text="Path :", font=('arial', 18), bd=18)
    Path_label.grid(row=2)
    Path_entry = Entry(Inputs_Frame, textvariable=Path,font=('arial', 14), width=30)
    Path_entry.grid(row=2, column=1)

    Discretization_label = Label(Inputs_Frame, text="Choices of discretization type :", font=('arial', 18), bd=18)
    Discretization_label.grid(row=3)
    Discretization_values=['Without Discretization', 'Equal frequency','Equal width','Based entropy']
    Discretization_Combo = Combobox(Inputs_Frame,values=Discretization_values, textvariable=Discretization, font=('arial', 15), width=20,state="readonly")
    Discretization_Combo.grid(row=3, column=1)
    Discretization_Combo.current(0)
    Discretization_Combo.bind("<<ComboboxSelected>>",Discretization_command)

    NumOFBins_label = Label(Inputs_Frame, text="Number of bins :", font=('arial', 18), bd=18)
    NumOFBins_label.grid(row=4)
    NumOFBins_entry= Entry(Inputs_Frame, font=('arial', 20), textvariable=NumOFBins, width=15,state='disabled')
    NumOFBins_entry.grid(row=4, column=1)

    Algorithm_label = Label(Inputs_Frame, text="Choices of Algorithms :", font=('arial', 18), bd=18)
    Algorithm_label.grid(row=6)
    Algorithm_values=['naive bayes classifier','ID3','KNN','K-MEANS']
    Algorithm_Combo = Combobox(Inputs_Frame,values=Algorithm_values, textvariable=Algorithm,font=('arial', 15), width=20,state="readonly")
    Algorithm_Combo.grid(row=6, column=1)
    Algorithm_Combo.current(0)
    Algorithm_Combo.bind("<<ComboboxSelected>>", Algorithm_command)


    NumOfNeighbors_label = Label(Inputs_Frame, text="Number of neighbors :", font=('arial', 18), bd=18)
    NumOfNeighbors_label.grid(row=7)
    NumOfNeighbors_entry= Entry(Inputs_Frame, font=('arial', 20), textvariable=NumOfNeighbors, width=15,state='disabled')
    NumOfNeighbors_entry.grid(row=7, column=1)

    result_label = Label(Inputs_Frame, text="", font=('arial', 18),fg="red")
    result_label.grid(row=8, columnspan=2)
    result_button = Button(Inputs_Frame, text="Run The Algorithm", font=('arial', 18),command=InputsForm_To_ResultForm, width=35,fg='purple')
    result_button.grid(row=9, columnspan=2, pady=20)

def Discretization_command(event):
    '''
    command function to the discretization combo box
    :return:nothing
    '''
    if Discretization.get()!='Without Discretization':
        NumOFBins_entry.configure(state="normal")
    else:
        NumOFBins.set("")
        NumOFBins_entry.configure(state="disabled")

def Algorithm_command(event):
    '''
    command function to the algorithm combo box
    :return:nothing
    '''
    if Algorithm.get() == 'KNN':
        NumOfNeighbors_entry.configure(state="normal")
        NumOfNeighbors_label['text']='Number of neighbors :'
    else:
        if Algorithm.get() == 'K-MEANS':
            NumOfNeighbors_entry.configure(state="normal")
            NumOfNeighbors_label['text'] = 'Number of clusters :'
        else:
            NumOfNeighbors.set("")
            NumOfNeighbors_entry.configure(state="disabled")


def InputsForm_To_ResultForm():
    '''
    function that check if the inputs of the user are propers
    :return:nothing
    '''
    if Algorithm.get() in ['KNN','K-MEANS']:
        if NumOfNeighbors.get() == '':
            NumOfNeighbors.set('2')
        else:
            try:
                num_neg = int(NumOfNeighbors.get())
                if num_neg <= 0:
                    result_label.configure(text="Number of neighbors field accept positive numbers")
                    return ''
            except ValueError:
                result_label.configure(text="Number of neighbors field accept just numbers")
                return ''

    if Path.get()=='':
        result_label.configure(text="Path field is empty")
    else:
        if os.path.exists(Path.get())==False:
            result_label.configure(text="The given path is not exist")
        else:
            if Discretization.get()!='Without Discretization':
                if NumOFBins.get()=='':
                    result_label.configure(text="Number of bins field is empty")
                else:
                    try:
                        num_bins=int(NumOFBins.get())
                        if num_bins <= 0:
                            result_label.configure(text="Number of bins field accept positive numbers")
                        else:
                            result_label.configure(text="")
                            Inputs_Frame.destroy()
                            ResultForm()
                    except ValueError:
                        result_label.configure(text="Number of bins field accept just numbers")
            else:
                result_label.configure(text="")
                Inputs_Frame.destroy()
                ResultForm()


def ResultForm_To_InputsForm():
    '''
    function that convert the frame from the main frame to the result frame
    :return:nothing
    '''
    Path.set("")
    NumOFBins.set("")
    Result_Frame.destroy()
    InputsForm()

def ResultForm():
    '''
    function that build a result frame that contain all the results for the inputs that user gives .
    :return:nothing
    '''
    global Result_Frame,label
    Result_Frame = Frame(window)
    Result_Frame.pack()

    if Discretization.get() != 'Without Discretization':
        bins=int(NumOFBins.get())
    else:
        bins=''

    try:
        results=Run_Algoritm(Path.get(),Algorithm.get(),Discretization.get(),bins,2 if NumOfNeighbors.get()=='' else int(NumOfNeighbors.get()))
        train_matrix=results['train']['Confusion Matrix']
        test_matrix=results['test']['Confusion Matrix']

        label1 = Label(Result_Frame, text="Confusion Matrix (Train)",font=('Arial',18, 'bold'))
        label1.grid(row=0)
        label2 = Label(Result_Frame, text=" ")
        label2.grid(row=1)

        lst = [('TP = '+str(train_matrix[0][0]),'FP = '+str(train_matrix[0][1])),('FN = '+str(train_matrix[1][0]),'TN = '+str(train_matrix[1][1]))]
        for i in range(2):
            for j in range(2):
                e = Entry(Result_Frame, width=20, fg='blue',font=('Arial', 16, 'bold'))
                e.grid(row=i+2, column=j)
                e.insert(END, lst[i][j])

        label3 = Label(Result_Frame, text="",font=('Arial',18, 'bold'))
        label3.grid(row=4)
        label4 = Label(Result_Frame, text="Confusion Matrix (Test)",font=('Arial',18, 'bold'))
        label4.grid(row=5)
        label5 = Label(Result_Frame, text=" ")
        label5.grid(row=6)

        lst = [('TP = '+str(test_matrix[0][0]),'FP = '+str(test_matrix[0][1])),('FN = '+str(test_matrix[1][0]),'TN = '+str(test_matrix[1][1]))]
        for i in range(2):
            for j in range(2):
                e2 = Entry(Result_Frame, width=20, fg='blue',font=('Arial', 16, 'bold'))
                e2.grid(row=i+7, column=j)
                e2.insert(END, lst[i][j])

        label6 = Label(Result_Frame, text=" ")
        label6.grid(row=9)
        label7 = Label(Result_Frame, text="Results",font=('Arial',18, 'bold'))
        label7.grid(row=10)
        label8 = Label(Result_Frame, text=" ")
        label8.grid(row=11)

        lst = [(' ','Accuracy','Precision','Recall','F-measure')]
        for i in ['train','test']:
            Accuracy="{0:.2f} %".format(results[i]['Accuracy'] *100)
            Precision="{0:.2f} %".format(results[i]['Precision'] *100)
            Recall="{0:.2f} %".format(results[i]['Recall'] *100)
            Measure="{0:.2f} %".format(results[i]['F-measure'] *100)
            lst.append((i,Accuracy,Precision,Recall,Measure))

        for i in range(3):
            for j in range(5):
                e2 = Entry(Result_Frame, width=20, fg='blue',font=('Arial', 16, 'bold'))
                e2.grid(row=i+12, column=j)
                e2.insert(END, lst[i][j])

        label9 = Label(Result_Frame, text=" ")
        label9.grid(row=16)

    except Exception as e:
        label = Label(Result_Frame, text=e.args[1], fg='Red',font=('Arial', 20, 'bold'))
        label.grid(row=16)

    Return_button = Button(Result_Frame, text="Return for the First Page", font=('arial', 18),command=ResultForm_To_InputsForm, width=35,fg='red')
    Return_button.grid(row=17,column=1, columnspan=2, pady=20)

def Run_Algoritm(Path,Algorithm,Discretization_type,NumOfBins,NumOfNeg):
    '''
    function that runs the algorithm that user choose and return the results of the user inputs and clean the files before that.
    :return:nothing
    '''
    pre = PreProcessing()

    test = pd.read_csv(Path+'\\test.csv')
    train = pd.read_csv(Path+'\\train.csv')
    struct = pre.read_structure(Path+'\\Structure.txt')

    pre.Clean_Data(train, struct,Discretization_type,NumOfBins)
    pre.Delete_Nan_Class_Row(test)
    pre.Fill_Nan_Values(test, struct)

    pre.Save_Data(train,Path, 'train')
    pre.Save_Data(test,Path, 'test')

    runner = BuildAlgorithm()
    train, test = runner.Convert_Strings_To_Numbers(Path)
    return runner.Run(Algorithm, train, test, Path,NumOfNeg)

#running the main frame
InputsForm()
window.mainloop()

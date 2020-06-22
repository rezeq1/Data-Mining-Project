import os
from   tkinter import *
from tkinter.ttk import Combobox

window=Tk()
window.title('Data Mining Project')
width =1060
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
Algorithm = StringVar()

def InputsForm():
    global Inputs_Frame, result_label,NumOFBins_entry
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

    result_label = Label(Inputs_Frame, text="", font=('arial', 18),fg="red")
    result_label.grid(row=7, columnspan=2)
    result_button = Button(Inputs_Frame, text="Run The Algorithm", font=('arial', 18),command=InputsForm_To_ResultForm, width=35,fg='purple')
    result_button.grid(row=8, columnspan=2, pady=20)

def Discretization_command(event):
    if Discretization.get()!='Without Discretization':
        NumOFBins_entry.configure(state="normal")
    else:
        NumOFBins.set("")
        NumOFBins_entry.configure(state="disabled")

def InputsForm_To_ResultForm():
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
    Path.set("")
    NumOFBins.set("")
    Result_Frame.destroy()
    InputsForm()

def ResultForm():
    global Result_Frame
    Result_Frame = Frame(window)
    Result_Frame.pack()

    Path_label = Label(Result_Frame, text="         The results", font=('arial', 25), bd=20)
    Path_label.grid(row=0)

    Return_button = Button(Result_Frame, text="Return for the First Page", font=('arial', 18),command=ResultForm_To_InputsForm, width=35,fg='purple')
    Return_button.grid(row=8, columnspan=2, pady=20)

InputsForm()
window.mainloop()

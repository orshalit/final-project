import PySimpleGUI as sg


sg.theme('Dark Blue 3')

layout = [[sg.Text('Choose an image to classify: ',font=("Helvetica", 30))],
          [sg.Input(), sg.FileBrowse(key='img')],
          [sg.Button('Classify My Pearl')]
]


window = sg.Window('Pearls Classifier', layout, size=(600, 600), auto_size_text=True, auto_size_buttons=True, margins=(30,30), finalize=True)

ImageToClassify = None
while True:
    event, values = window.read()
    if event in (None, 'Exit'):
        break
    try:
        ImageToClassify = str(values["img"])
    except:
        pass
        #ImageToClassify = None
    if event == 'Classify My Pearl':
        if ImageToClassify == '':  #string of the path of the image
            sg.popup('Please choose an image!!!')
        else:
            print("here is the part to classify the image")
window.close()
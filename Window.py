from tkinter import *
from PIL import Image, ImageTk

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.card_entry = []
        self.init_window()

    def init_window(self):
        self.master.title("Black Jack GUI")
        self.pack(fill=BOTH, expand=1)

        # Make menu
        self.create_menu()

        # edit = Menu(menu)
        # edit.add_command(label="Show Img", command=self.showImg)
        # edit.add_command(label="Show Img", command=self.showTxt)
        # menu.add_cascade(label="Edit", menu=edit)

        # Text entry format
        self.create_card_text()
        Button(self.master, text='Return Result', width=6, command=self.analyze).grid(row=3, column=0, sticky=W)


    def create_menu(self):
        menu = Menu(self.master)
        self.master.config(menu=menu)
        file = Menu(menu)

        file.add_command(label='Save')
        file.add_command(label="Exit", command=self.client_exit)
        menu.add_cascade(label="File", menu=file)

    def create_card_text(self):
        self.card_entry[0] = Entry(self.master, width=10, bg='white')
        self.card_entry[0].grid(row=2, column=0, sticky=W)

        self.card_entry[1] = Entry(self.master, width=10, bg='white')
        self.card_entry[1].grid(row=2, column=1, sticky=W)

        self.card_entry[2] = Entry(self.master, width=10, bg='white')
        self.card_entry[2].grid(row=2, column=2, sticky=W)

        self.card_entry[3] = Entry(self.master, width=10, bg='white')
        self.card_entry[3].grid(row=2, column=3, sticky=W)

    def client_exit(self):
        exit()

    def analyze(self):
        cards = []
        for i, entry in enumerate(self.card_entry):
            cards[i] = entry.get()
        print(cards)

    def showImg(self):
        load = Image.open('filename.png')
        render = ImageTk.PhotoImage(load)

        img = Label(self, image=render)
        img.image = render
        img.place(x=0, y=0)

    def showTxt(self):
        text = Label(self, text='Some Text')
        text.pack()


# Root window, create
root = Tk()
root.geometry("400x300")

#
app = Window(root)
root.mainloop()

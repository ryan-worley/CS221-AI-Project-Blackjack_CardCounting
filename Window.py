from tkinter import *
from PIL import Image, ImageTk
from collections import *
import random as rand
import pickle

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.card_entry = []
        self.card_values = ('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A')
        self.suits = ('C', 'S', 'H', 'D')
        self.image_files = self.create_image_dict()
        self.counts = [i for i in range(-10, 11)]
        self.pi, self.V = self.loadPolicy()
        self.img = []
        self.count_input = None
        self.V_current = None
        self.pi_current = None
        self.count = None
        self.cards = None
        self.output = ['', '']
        self.warning = ''
        self.background = Label(self.master, image=ImageTk.PhotoImage(Image.open('./GUI/TB2.jpg')))
        self.background.image = ImageTk.PhotoImage(Image.open('./GUI/TB2.jpg'))
        self.init_window()


    def init_window(self):
        self.master.title("Black Jack GUI")
        self.master.config(borderwidth=3, relief='ridge')

        # # Change color to darker green
        # self.background.place(x=0, y=0, relwidth=1, relheight=1)

        # Make menu
        self.create_menu()

        # Text entry format
        self.create_card_text()

        # Create button that takes in results, creates dict of them
        Button(self.master, text='Return Result', width=15, command=self.analyze).grid(row=7, column=0, columnspan=2,
                                                                                       padx=5, pady=10,
                                                                                       sticky='n')

    def analyze(self):
        # Get the inputs from the GUI
        self.cards = self.getCards()
        self.count = self.getCount()
        cards = self.cards.values()


        # Show corresponding images
        player_image, dealer_image = self.get_card_image()
        self.show_card_image(player_image, dealer_image)

        # Get State given all cards inputted
        state, pvalue, dcard = self.getState()

        FLAG = False
        warning = []
        if pvalue == 21:
            self.V_current = 1.5
            self.pi_current = 'Winna Winna'
            FLAG = True
        if pvalue > 21:
            self.V_current = -1
            self.pi_current = 'Busted, no Action available'
            FLAG = True
        elif len(player_image) == 1:
            self.V_current = 0
            self.pi_current = "Please enter more player cards"
            FLAG = True
        elif not dcard:
            self.V_current = 0
            self.pi_current = "Please Enter Dealer Shown Card"
            FLAG = True
        elif not self.count:
            self.V_current = 0
            self.pi_current = 'Please Enter a Count Value'
            FLAG = True

        if FLAG:
            self.displayState()
        else:
            print(state, 'state')
            self.V_current = self.V[state]
            self.pi_current = self.pi[state]
            self.displayState()
            print(self.V_current, 'V')
            print(self.pi_current, 'pi')

    def displayState(self):
        if self.output:
            self.clearLabels()
        self.output[1] =  Label(self.master, text='%#.5G' % self.V_current)
        self.output[0] = Label(self.master, text=str(self.pi_current))
        self.output[0].grid(column=1, row=11)
        self.output[1].grid(column=1, row=12)

    def clearLabels(self):
        for label in self.output:
            if label != '':
                label.destroy()

    def getState(self):
        player_cards = [card for player, card in self.cards.items() if player[0] == 'p']
        dealer_card = [card for player, card in self.cards.items() if player[0] == 'd']
        player_state, pvalue = self.player_state(player_cards)
        dealer_state, _ = self.player_state(dealer_card)
        state = (player_state, dealer_state, int(self.count))
        return state, pvalue, dealer_state

    def getCards(self):
        cards = defaultdict(str)
        order = ['p1', 'p2', 'p3', 'p4', 'd1']
        for i, entry in enumerate(self.card_entry):
            if entry.get() != '':
                cards[order[i]] = entry.get()
        return cards

    def getCount(self):
        return self.count_input.get()

    def clearImage(self):
        if self.img:
            for image in self.img:
                image.config(image='')

    def player_state(self, cards):
        double = ''
        split = ''
        acecounter = 0
        value = 0
        aces = ''
        if not cards:
            return '', 0

        if len(cards) == 1:
            try:
                value += int(cards[0])
            except:
                if cards[0] == 'A':
                    acecounter += 1
                    value += 11
                elif cards[0] in ('T', 'J', 'Q', 'K'):
                    value += 10
            return value, value

        if len(cards) == 2:
            double = 'D'
            if cards[0] == cards[1]:
                split = 'S'

        for card in cards:
            try:
                value += int(card)
            except:
                if card == 'A':
                    acecounter += 1
                    value += 11
                elif card in ('T', 'J', 'Q', 'K'):
                    value += 10

        while value > 21 and acecounter > 0:
            acecounter -= 1
            value -= 10
        if acecounter == 1:
            aces = 'A'
        state = str(value) + '*' + aces + double + split
        return state, value

    def loadPolicy(self):
        policy = defaultdict(str)
        V = defaultdict(float)
        for count in self.counts:
            current_pi = pickle.load(open('./policy/' + 'Count {} Policy.pkl'.format(count), 'rb'))
            policy.update(current_pi)
            current_V = pickle.load(open('./policy/' + 'Count {} V.pkl'.format(count), 'rb'))
            V.update(current_V)
        return policy, V

    def create_image_dict(self):
        imagefiles = defaultdict(list)
        init = 'card'
        for card in self.card_values:
            for suit in self.suits:
                imagefiles[card].append(init + card + suit)
        return imagefiles

    def get_card_image(self):
        image_player = defaultdict(str)
        image_dealer = ''
        for key, card in self.cards.items():
            if key[0] == 'p':
                image_player[key[1]] = rand.choice(self.image_files[card])
            else:
                image_dealer = rand.choice(self.image_files[card])
        return image_player, image_dealer

    def show_card_image(self, playerimage, dealerimage):
        self.clearImage()
        path = './GUI/'
        extension = '.png'
        col = 12
        index = 0
        self.img = []

        for num, image in playerimage.items():
            load = Image.open(path + image + extension)
            load.resize((200, 250))
            render = ImageTk.PhotoImage(load)
            self.img.append(Label(self.master, image=render))
            self.img[index].image = render
            self.img[index].grid(row=1, column=col, columnspan=1, rowspan=4, padx=5, pady=5, sticky='nw')
            col += 1
            index += 1

        index = len(playerimage)

        if playerimage.keys():
            Label(self.master, text='Player:', width=25, anchor='e').grid(row=1, column=11, sticky='e', padx=5)

        if dealerimage:
            load = Image.open(path + dealerimage + extension)
            load.resize((200, 250))
            render = ImageTk.PhotoImage(load)

            img = Label(self.master, image=render)
            img.image = render
            img.grid(row=4, column=12, rowspan=4, padx=5, pady=5, sticky='nw')
            Label(self.master, text='Dealer:', anchor='e', width=25).grid(row=4, column=11, sticky='e', padx=5)

    def create_menu(self):
        menu = Menu(self.master)
        self.master.config(menu=menu)
        file = Menu(menu)

        file.add_command(label="Exit", command=self.client_exit)
        menu.add_cascade(label="File", menu=file)

    def create_card_text(self):
        pad = 5
        Label(self.master, text='State Initial Inputs: Card = (2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K, A)'
              ).grid(row=0, column=0, columnspan=20, rowspan=1, padx=5, pady=5, sticky='w')

        Label(self.master, text='Player Card 1:').grid(row=1, column=0, sticky='e')
        self.card_entry.append((Entry(self.master, width=12, bg='white')))
        self.card_entry[0].grid(row=1, column=1, padx=pad, pady=pad)

        Label(self.master, text='Player Card 2:').grid(row=2, column=0, sticky='e')
        self.card_entry.append(Entry(self.master, width=12, bg='white'))
        self.card_entry[1].grid(row=2, column=1, padx=pad, pady=pad)

        Label(self.master, text='Player Card 3:').grid(row=3, column=0, sticky='e')
        self.card_entry.append(Entry(self.master, width=12, bg='white'))
        self.card_entry[2].grid(row=3, column=1, padx=pad, pady=pad)

        Label(self.master, text='Player Card 4:').grid(row=4, column=0, sticky='e')
        self.card_entry.append(Entry(self.master, width=12, bg='white'))
        self.card_entry[3].grid(row=4, column=1, padx=pad, pady=pad)

        Label(self.master, text='Shown Dealer Card:').grid(row=5, column=0, sticky='e')
        self.card_entry.append(Entry(self.master, width=12, bg='white'))
        self.card_entry[4].grid(row=5, column=1, padx=pad, pady=pad)

        Label(self.master, text='Card Count:').grid(row=6, column=0, sticky='e')
        self.count_input = Entry(self.master, width=12, bg='white')
        self.count_input.grid(row=6, column=1, padx=pad, pady=pad)

        Label(self.master, text='Optimum Policy:').grid(row=11, column=0, sticky='e')
        Label(self.master, text='Expected Reward:').grid(row=12, column=0, sticky='e')

    def client_exit(self):
        exit()


# Root window, create
root = Tk()
root.geometry("700x300")

app = Window(root)
root.mainloop()

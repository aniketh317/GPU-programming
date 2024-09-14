# Based on code from https://github.com/Nohclu/Sleeping-Barber-Python-3.6-/blob/master/barber.py
import time
import random
import threading
from queue import Queue

CUSTOMERS_SEATS = 15        # Number of seats in BarberShop
BARBERS = 3                # Number of Barbers working
EVENT = threading.Event()   # Event flag, keeps track of Barber/Customer interactions
global Earnings
global SHOP_OPEN

earnings_lock = threading.Lock() #Lock for earnings update
customerQueue_lock = threading.Lock() #Lock for accesing queue

class Customer(threading.Thread):       # Producer Thread
    def __init__(self, queue):          # Constructor passes Global Queue (all_customers) to Class
        threading.Thread.__init__(self)
        self.queue = queue
        self.rate = self.what_customer()

    @staticmethod
    def what_customer():
        customer_types = ["adult", "senior", "student", "child"]
        customer_rates = {"adult": 16,
                          "senior": 7,
                          "student": 10,
                          "child": 7}
        t = random.choice(customer_types)
        print(t + " rate.")
        return customer_rates[t]

    def run(self):
        with customerQueue_lock:
            if not self.queue.full():  # Check queue size
                EVENT.set()  # Sets EVENT flag to True i.e. Customer available in the Queue
                EVENT.clear()  # A lerts Barber that their is a Customer available in the Queue
            else:
                # If Queue is full, Customer leaves.
                print("Queue full, customer has left.")

    def trim(self):
        print("Customer haircut started.")
        a = 3 * random.random()  # Retrieves random number.
        # TODO execute the time sleep function with a, which simulates the time it takes for a barber to give a haircut.
        time.sleep(a)
        payment = self.rate
        # Barber finished haircut.
        print("Haircut finished. Haircut took {}".format(a))
        global Earnings
        with earnings_lock:
            Earnings += payment


class Barber(threading.Thread):     # Consumer Thread
    def __init__(self, queue):      # Constructor passes Global Queue (all_customers) to Class
        threading.Thread.__init__(self)
        # TODO set this class's queue property to the passed value
        self.queue = queue
        self.sleep = True   # No Customers in Queue therefore Barber sleeps by default

    def is_empty(self):  # Simple function that checks if there is a customer in the Queue and if so
        if self.queue.empty():
            self.sleep = True   # If nobody in the Queue Barber sleeps.
        else:
            self.sleep = False  # Else he wakes up.
        print(f"------------------\nBarber --{self.name} sleep {self.sleep}\n------------------")

    def run(self):
        global SHOP_OPEN
        while SHOP_OPEN:
            customerQueue_lock.acquire() # Lock Acquire
            while self.queue.empty() and SHOP_OPEN:
                # Waits for the Event flag to be set, Can be seen as the Barber Actually sleeping.
                customerQueue_lock.release() # Releasing lock since going into wait
                print(f"Barber --{self.name} is sleeping...")
                EVENT.wait()
                if not SHOP_OPEN:
                    break
                print(f"Barber --{self.name} is being woken up")
                customerQueue_lock.acquire()
            if not SHOP_OPEN:
                break
            print("Barber is awake.")
            customer = self.queue
            self.is_empty()
            if self.sleep:
                customerQueue_lock.release()
                continue
            # FIFO Queue So first customer added is gotten.
            customer = customer.get()
            customerQueue_lock.release() #Releasing lock
            customer.trim()  # Customers Hair is being cut
            customer = self.queue
            
            # TODO use the task_done function to complete cutting the customer's hair
            with customerQueue_lock:
                customer.task_done() # Indicating that a task has been done
            print(self.name)    # Which Barber served the Customer


def wait():
    time.sleep(1 * random.random())


if __name__ == '__main__':
    Earnings = 0
    SHOP_OPEN = True
    barbers = []
    all_customers = Queue(CUSTOMERS_SEATS)  # A queue of size Customer Seats

    for b in range(BARBERS):
        # TODO Pass the all_customers Queue to the Barber constructor
        b = Barber(all_customers)
        # Makes the Thread a super low priority thread allowing it to be terminated easier
        b.daemon = True
        b.start()   # Invokes the run method in the Barber Class
        # Adding the Barber Thread to an array for easy referencing later on.
        barbers.append(b)
    for c in range(10):  # Loop that creates infinite Customers
        print("----")
        # Simple Tracker too see the qsize (NOT RELIABLE!)
        with customerQueue_lock:
            print(f"Customer queue size is -- {all_customers.qsize()}")
        wait()
        c = Customer(all_customers)  # Passing Queue object to Customer class
        with customerQueue_lock:
            all_customers.put(c)    # Puts the Customer Thread in the Queue
        # TODO Invoke the run method in the Customer Class
        c.run()
    all_customers.join()    # Terminates all Customer Threads
    earnings = str(int(Earnings))
    print("Barbers payment total:"+earnings)
    SHOP_OPEN = False
    for i in barbers:
        EVENT.set() # Event being set for one last time so that the barber processes exit
        i.join()    # Terminates all Barbers
        # Program hangs due to infinite loop in Barber Class, use ctrl-z to exit.
    EVENT.clear() # Resetting the event flag
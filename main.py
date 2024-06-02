import cv2
import imutils
import pytesseract
import tkinter as tk
from PIL import Image, ImageTk   
from tkinter.ttk import *
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
from collections import Counter
from sklearn.cluster import KMeans
import mysql.connector
from tkinter import *
from tkinter import filedialog
from tkinter.messagebox import *
from PIL import Image, ImageTk
import csv
import re
from datetime import datetime

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import pandas as pd
from pandastable import Table, TableModel
from datetime import datetime, date


# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


BUTTON_BACK = '#364156'
BUTTON_FORG = 'white'
LABEL_BACK = '#CDCDCD'
BACK = '#CDCDCD'

# -----------  DATA UPDATE  -----------
cols = [0, 1, 2]
df = pd.read_excel('vish.xlsx', usecols=cols)

# -----------  TKINTER INIT  -----------
root = Tk()
root.geometry('1100x900')
root.title('License Plate Logging System')
root.configure(background=BACK)

mail_content = '''Hello,
This is a mail from Automated License Plate Recognition System.
In this mail we are sending the excel file of License Plate.
Thank You
'''

s = Style()
s.theme_create("LIGHT_MODE", parent="alt", settings={
    "TNotebook": {"configure": {"tabmargins": [140, 0, 2, 0], "background": "#23272A"}},
    "TNotebook.Tab": {"configure": {"padding": [80, 10], "font": ('URW Gothic L', '11', 'bold'), "background": "#fff", "foreground": "#23272A"},
                      "map": {"background": [("selected", '#CDCDCD')],
                              "expand": [("selected", [1, 1, 1, 0])]}}
})

s.theme_create("DARK_MODE", parent="alt", settings={
    "TNotebook": {"configure": {"tabmargins": [140, 0, 2, 0], "background": "#23272A"}},
    "TNotebook.Tab": {"configure": {"padding": [80, 10], "font": ('URW Gothic L', '11', 'bold'), "background": "#23272A", "foreground": '#fff'},
                      "map": {"background": [("selected", '#23272A')],
                              "expand": [("selected", [1, 1, 1, 0])]}}
})

s.theme_use("LIGHT_MODE")

heading = Label(root, text="License Plate Detection System", font=('arial', 20, 'bold'))
heading.configure(background='#eee', foreground='#364156')
heading.pack()

# -----------  TABS  -----------
TABS = Notebook(root)

image_tab = Frame(TABS)
TABS.add(image_tab, text="Image")
TABS.pack(expand=1, fill="both")


details_tab = Frame(TABS)
TABS.add(details_tab, text="Details")
TABS.pack(expand=1, fill="both")

about_tab = Frame(TABS)
TABS.add(about_tab, text="About")
TABS.pack(expand=1, fill="both")

def detect_color(image, k=4):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(pixels)
    label_counts = Counter(labels)
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    dominant_color = [int(x) for x in dominant_color]
    return dominant_color

def create_table():
    try:
        # Establish database connection
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="vehicle_info"
        )
        cursor = conn.cursor()

        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_info (
                id INT AUTO_INCREMENT PRIMARY KEY,
                license_plate VARCHAR(255),
                owner VARCHAR(255),
                registration_status VARCHAR(255),
                car_color_hex VARCHAR(255),
                car_make VARCHAR(255),
                car_model VARCHAR(255),
                state VARCHAR(255), 
                phone_number VARCHAR(255),
                tinted ENUM('true', 'false'), 
                start_date DATE, 
                end_date DATE
            )
        """)

        print("Table created successfully")
    except mysql.connector.Error as err:
        print("Error: ", err)
    finally:
        # Close the database connection
        if conn.is_connected():
            cursor.close()
            conn.close()


def save_to_db(license_plate, owner, registration_status,  car_color_hex, car_make, car_model, state, phone_number, tinted, start_date, end_date):
    try:
        # Establish database connection
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="vehicle_info"
        )
        cursor = conn.cursor()
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        # Insert data into the table
        sql = """
            INSERT INTO vehicle_info (
                license_plate, owner, registration_status,  car_color_hex, car_make, car_model, state, phone_number, tinted, start_date, end_date
                ) VALUES (%s, %s,  %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            license_plate, owner, registration_status,  car_color_hex, car_make, car_model, 
            state, phone_number, tinted, start_date, end_date
        )
        cursor.execute(sql, values)

        # Commit the transaction
        conn.commit()
        print("Data saved successfully")
    except mysql.connector.Error as err:
        print("Error: ", err)
    finally:
        # Close the database connection
        if conn.is_connected():
            cursor.close()
            conn.close()


def send_mail():
    r = askokcancel(title="Mail Excel", message="Do you want to mail the excel file to admin")
    if r:
        sendermail = "ogunburebusayo.j@gmail.com"
        recivermail = "ogunburebusayo.j@gmail.com"
        password = "jpkisglohidadtlc"

        try:
            message = MIMEMultipart()
            message['From'] = sendermail
            message['To'] = recivermail
            message['Subject'] = 'A test mail sent by Python. It has an attachment.'
            message.attach(MIMEText(mail_content, 'plain'))
            attach_file_name = 'vish.xlsx'
            attach_file = open(attach_file_name, 'rb')  # Open the file as binary mode
            payload = MIMEBase('application', 'octate-stream')
            payload.set_payload((attach_file).read())
            encoders.encode_base64(payload)  # encode the attachment
            # add payload header with filename
            payload.add_header('Content-Disposition', 'attachment; filename="vish.xlsx"')
            # payload.add_header('Content-Decomposition', 'attachment', filename="LP.xlsx")
            message.attach(payload)
            # Create SMTP session for sending the mail
            session = smtplib.SMTP('smtp.gmail.com', 587)  # use gmail with port
            session.starttls()  # enable security
            session.login(sendermail, password)  # login with mail_id and password
            text = message.as_string()
            session.sendmail(sendermail, recivermail, text)
            session.quit()
            showinfo(title="Mail Sent", message="Mail has been successfully sent")
            print("success")
        except Exception as e:
            print("Failed", e)
            showwarning(title="Mail Not Sent", message="Mail has not been sent. Check your connectivity.")
        # exit()

def change_theme():
    if change_theme['text'] in "DARK MODE":
        change_theme.configure(text="LIGHT MODE")
        heading.configure(background='#000000', foreground='#FFF')
        root.configure(background='#000000')
        image_tab.configure(background='#000000')
        video_tab.configure(background='#000000')
        details_tab.configure(background='#000000')
        about_tab.configure(background='#000000')
        s.theme_use("DARK_MODE")  # 00CED1
        classify_b.configure(background='#000000', foreground='#00FFFF')
        send_mail_button.configure(background='#000000', foreground='#00FFFF')
        change_theme.configure(background='#000000', foreground='#00FFFF')
        upload.configure(background='#000000', foreground='#00FFFF')
        upload_video.configure(background='#000000', foreground='#00FFFF')
        display.configure(background="#CDCDCD", foreground="#000000")
        display_video.configure(background="#CDCDCD", foreground="#000000")

        image_tab.update()
    else:
        change_theme.configure(text="DARK MODE")
        heading.configure(background='#eee', foreground='#364156')
        root.configure(background='#CDCDCD')
        image_tab.configure(background='#CDCDCD')
        video_tab.configure(background='#CDCDCD')
        details_tab.configure(background='#CDCDCD')
        about_tab.configure(background='#CDCDCD')
        s.theme_use("LIGHT_MODE")
        classify_b.configure(foreground='#364156', background='#CDCDCD')
        send_mail_button.configure(foreground='#364156', background='#CDCDCD')
        change_theme.configure(foreground='#364156', background='#CDCDCD')
        upload.configure(foreground='#364156', background='#CDCDCD')
        upload_video.configure(foreground='#364156', background='#CDCDCD')
        display.configure(background='#364156', foreground='#fff')
        display_video.configure(background='#364156', foreground='#fff')

def close_window():
    root.destroy()

def clear_dets():
    list1.delete(0, END)
    list1.insert(END, "Showing All License Plates Detected Till Date")

def close_root():
    root.destroy()


def read_vehicle_data(csv_file):
    vehicle_data = {}
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            license_plate = row['License Plate']
            vehicle_data[license_plate] = row
    return vehicle_data


def find_vehicle_details(vehicle_data, license_plate):
    return vehicle_data.get(license_plate, None)

def upload_image():
    try:
        # Get the file path from the dialog
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((root.winfo_width() / 2.25), (root.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        # Create a label to display the image
        car_label.configure(image=im)
        car_label.image = im

        # Read and preprocess the image
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (500, 400))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 13, 15, 15)

        # Edge detection and contour finding
        edged = cv2.Canny(gray, 30, 200)
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        screenCnt = None

        # Finding the license plate contour
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is None:
            print("No contour detected")
        else:
            cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
            new_image = cv2.bitwise_and(img, img, mask=mask)
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            Cropped = gray[topx:bottomx+1, topy:bottomy+1]
            text = pytesseract.image_to_string(Cropped, config='--psm 11')
            
            # Clean up the detected text
            cleaned_text = re.sub(r'\W+', '', text).strip().upper()
            print("Detected license plate number is:", cleaned_text)

            # Detect car color
            car_color = detect_color(img)
            car_color_hex = '#%02x%02x%02x' % tuple(car_color)
            car_color_rgb = ','.join(map(str, car_color))
            print("Detected car color (RGB):", car_color)
            print("Detected car color (Hex):", car_color_hex)
            
            
              # Read CSV file and match the detected license plate
            vehicle_data = pd.read_csv('vehicle_data.csv')
            matched_vehicle = vehicle_data[vehicle_data['License Plate'] == cleaned_text]

            if not matched_vehicle.empty:
                owner = matched_vehicle['Owner Name'].values[0]
                registration_status = matched_vehicle['Registration Status'].values[0]
                car_make = matched_vehicle['Car Make'].values[0]
                car_model = matched_vehicle['Car Model'].values[0]
                state = matched_vehicle['State'].values[0]
                phone_number = matched_vehicle['Phone Number'].values[0]
                start_date = matched_vehicle['Start Date'].values[0]
                end_date = matched_vehicle['End Date'].values[0]
                tinted = matched_vehicle['Tinted'].values[0]

                # Convert numpy.int64 to Python int if necessary
                if isinstance(phone_number, np.int64):
                    phone_number = int(phone_number)
                if isinstance(start_date, np.int64):
                    start_date = int(start_date)
                if isinstance(end_date, np.int64):
                    end_date = int(end_date)
                if isinstance(tinted, np.bool_):
                    tinted = bool(tinted)

                print("Owner:", owner)
                print("Registration Status:", registration_status)
                print("Car Make:", car_make)
                print("Car Model:", car_model)
                print("State:", state)
                print("Phone Number:", phone_number)
                print("Start Date:", start_date)
                print("End Date:", end_date)
                print("Tinted:", tinted)
            else:
                print("Owner: Not Found")
                print("Registration Status: Not Found")
                print("Car Make: Not Found")
                print("Car Model: Not Found")
                print("State: Not Found")
                print("Phone Number: Not Found")
                print("Start Date: Not Found")
                print("End Date: Not Found")
                print("Tinted: Not Found")
            create_table()
            save_to_db(cleaned_text, owner, registration_status, car_color_hex, car_make, car_model,
                    state, phone_number, tinted, start_date, end_date)

        
            color_frame = np.zeros((200, 200, 3), np.uint8)
            car_color_bgr = (car_color[2], car_color[1], car_color[0])
            color_frame[:] = car_color_bgr

            # Resize images for display
            img = cv2.resize(img, (500, 300))
            Cropped = cv2.resize(Cropped, (400, 200))

  
            detected_lp.config(text=f"License Plate: {cleaned_text}")
            detected_owner.config(text=f"Owner:{owner}")
            detected_registration.config(text=f"Registration Status:{registration_status}")
            detected_color.configure(text=f"Car Color Hex:{car_color_hex}")
            detected_make.configure(text=f"Car Make:{car_make}")
            detected_model.configure(text=f"Car Model:{car_model}")
            detected_state.configure(text=f"State:{state}")
            detected_phone.configure(text=f"Phone:{phone_number}")
            detected_tinted.configure(text=f"Tinted:{tinted}")
            detected_start.configure(text=f"Start Date:{start_date}")
            detected_end.configure(text=f"End Date:{end_date}")
           

            # Display the images
            cv2.imshow('Car', img)
            cv2.imshow('License Plate', Cropped)
            cv2.imshow('Car Color', color_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", e)
    

def display_about():
    about_text = """
    Automated License Plate Recognition System
    
    AutoTrack is an automated License Plate Recognition (LPR) system designed to detect, identify,
    and log vehicle information from images. Utilizing computer vision and machine learning techniques, 
    AutoTrack extracts license plate numbers, detects vehicle colors, and matches the information 
    against a database of registered vehicles. The system also sends email notifications with relevant 
    data and maintains a MySQL database of vehicle details.
    
    Version: 1.0
    Developed by: Bussyboo
    """
    showinfo("About", about_text)


car_label = Label(image_tab)
car_label.grid(row=0, column=0, padx=10, pady=10)



detected_lp = Label(details_tab, text="License Plate: ")
detected_lp.grid(row=0, column=0, padx=10, pady=10)

detected_owner = Label(details_tab, text="Owner: ")
detected_owner.grid(row=0, column=0, padx=10, pady=10)

detected_registration = Label(details_tab, text="Registration Status: ")
detected_registration.grid(row=0, column=0, padx=10, pady=10)

detected_color = Label(details_tab, text="Color: ")
detected_color.grid(row=1, column=0, padx=10, pady=10)

detected_make = Label(details_tab, text="Make: ")
detected_make.grid(row=2, column=0, padx=10, pady=10)

detected_model = Label(details_tab, text="Model: ")
detected_model.grid(row=3, column=0, padx=10, pady=10)

detected_state = Label(details_tab, text="State: ")
detected_state.grid(row=3, column=0, padx=10, pady=10)

detected_phone = Label(details_tab, text="Phone: ")
detected_phone.grid(row=3, column=0, padx=10, pady=10)

detected_tinted = Label(details_tab, text="Tinted: ")
detected_tinted.grid(row=3, column=0, padx=10, pady=10)

detected_start = Label(details_tab, text="Start Date: ")
detected_start.grid(row=3, column=0, padx=10, pady=10)

detected_end = Label(details_tab, text="End Date: ")
detected_end.grid(row=3, column=0, padx=10, pady=10)


about_button = Button(about_tab, text="About", command=display_about, padx=10, pady=5)
about_button.grid(row=0, column=0, padx=10, pady=10)

classify_b = Button(root, text="Detect License Plate", command=upload_image, padx=10, pady=5)
classify_b.pack(side=TOP, pady=10)

send_mail_button = Button(root, text="Send Mail", command=send_mail, padx=10, pady=5)
send_mail_button.pack(side=TOP, pady=10)

change_theme = Button(root, text="DARK MODE", command=change_theme, padx=10, pady=5)
change_theme.pack(side=TOP, pady=10)

exit_button = Button(root, text="Exit", command=close_root, padx=10, pady=5)
exit_button.pack(side=TOP, pady=10)

root.mainloop()

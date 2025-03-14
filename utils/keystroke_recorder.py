import tkinter as tk
from tkinter import messagebox
import cv2
import threading
import time
import csv
import os
import tkinter.font as tkFont
from tkinter import ttk
from typing_data import typing_data

class TypingTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Typing Test")
        self.root.geometry("800x600")

        # Set a default font for the entire application
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(size=14)  # Increase the font size

        self.texts_to_type = typing_data # Use content from typing_data
        self.current_text_index = 0
        self.all_timings = [[] for _ in self.texts_to_type]
        self.media_recorder = None
        self.recording = False

        self.create_menu()
        self.create_widgets()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

    def show_about(self):
        messagebox.showinfo("About", "Typing Test Application\nVersion 1.0")

    def create_widgets(self):
        self.frame = tk.Frame(self.root, padx=10, pady=10, relief=tk.RAISED, borderwidth=2, bg="#e0e0e0")
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.video_label = tk.Label(self.frame, font=("Arial", 14, "bold"), bg="#e0e0e0", justify=tk.LEFT)
        self.video_label.pack(anchor=tk.W, pady=5)

        # self.label = tk.Label(self.frame, font=("Arial", 14), bg="#e0e0e0", justify=tk.LEFT)
        # self.label.pack(anchor=tk.W, pady=5)

        self.display_text = tk.Text(self.frame, height=5, wrap=tk.WORD, state=tk.DISABLED, bg="#f0f0f0", font=("Arial", 15))
        self.display_text.pack(fill=tk.BOTH, expand=True, pady=5)

        self.typing_area = tk.Text(self.frame, height=5, wrap=tk.WORD, bg="#ffffff", font=("Arial", 15))
        self.typing_area.pack(fill=tk.BOTH, expand=True, pady=5)
        self.typing_area.bind("<KeyRelease>", self.log_key)

        self.button_frame = tk.Frame(self.frame, bg="#e0e0e0")
        self.button_frame.pack(fill=tk.X, pady=5)

        # Use default tkinter buttons without custom styles
        self.start_btn = tk.Button(self.button_frame, text="Step 1: Start Recording", command=self.start_recording)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(self.button_frame, text="Step 3: Stop Recording", command=self.stop_recording)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = tk.Button(self.button_frame, text="Clear", command=self.clear_text)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.back_btn = tk.Button(self.button_frame, text="Back", command=self.previous_text)
        self.back_btn.pack(side=tk.LEFT, padx=5)

        self.next_btn = tk.Button(self.button_frame, text="Next", command=self.next_text)
        self.next_btn.pack(side=tk.LEFT, padx=5)

        self.indicator_label = tk.Label(self.frame, text="", font=("Arial", 10), bg="#e0e0e0")
        self.indicator_label.pack(anchor=tk.E, pady=5)

        self.frequency_label = tk.Label(self.frame, font=("Arial", 12), bg="#e0e0e0", justify=tk.LEFT, wraplength=400)
        self.frequency_label.pack(anchor=tk.W, pady=5)

        self.update_text_display()

        self.display_text.tag_config("correct", foreground="blue")  # Set color for correct letters
        self.display_text.tag_config("error", foreground="red")  # Set color for error letters

    def update_text_display(self):
        video_file = f'videos/video_{self.current_text_index}.mp4'  # Save video in 'videos' folder
        csv_file = f'labels/video_{self.current_text_index}.csv'      # Save CSV in 'labels' folder
        text_file = f'labels/video_{self.current_text_index}.txt'     # Save TXT in 'labels' folder
        files_exist = os.path.exists(video_file) and os.path.exists(csv_file) and os.path.exists(text_file)

        # Update label with tick mark if files exist
        tick_mark = "✔" if files_exist else ""
        tick_color = "green" if files_exist else "black"
        self.video_label.config(text=f"Text {self.current_text_index + 1}: {tick_mark}", fg=tick_color)
        
        self.display_text.config(state=tk.NORMAL)
        self.display_text.delete(1.0, tk.END)
        self.display_text.insert(tk.END, self.texts_to_type[self.current_text_index])
        self.display_text.config(state=tk.DISABLED)
        self.typing_area.delete(1.0, tk.END)
        self.indicator_label.config(text=f"Text {self.current_text_index + 1} of {len(self.texts_to_type)}")

        # Call the new method to display letter frequency
        self.display_letter_frequency(self.texts_to_type[self.current_text_index])

        # Update to show the current lesson title and intro
        current_lesson = typing_data[self.current_text_index]
        # self.label.config(text=f"Video {self.current_text_index}")  # Update label with title and intro

    def display_letter_frequency(self, text):
        frequency = {char: text.count(char) for char in set(text) if char.isalpha()}  # Count frequency in one line
        frequency_display = ", ".join(f"{char}: {count}" for char, count in frequency.items())
        self.frequency_label.config(text=f"Letter Frequency: \n{frequency_display}")  # Update the label with frequency information

    def log_key(self, event):
        if not self.recording:
            return
        key = event.keysym.lower()

        if key == 'shift_l' or key == 'shift_r' or key == 'caps_lock':
            key = 'shift'
    
        self.all_timings[self.current_text_index].append((key, self.frame_count))
        print(f"Key: {key}, Frame Count: {self.frame_count}")

        # Highlight the typed text
        typed_text = self.typing_area.get(1.0, tk.END).strip()  # Get the current typed text
        original_text = self.texts_to_type[self.current_text_index]  # Get the original text
        self.display_text.config(state=tk.NORMAL)  # Enable editing to apply tags

        # Clear previous highlights
        self.display_text.delete(1.0, tk.END)  # Clear the display text
        self.display_text.insert(tk.END, original_text)  # Insert the original text

        # Highlight the typed text
        # Text index in tkinter https://tkdocs.com/tutorial/text.html#modifying
        for i, char in enumerate(typed_text):
            if i < len(original_text):
                if char == original_text[i]:
                    self.display_text.tag_add("correct", f"1.0+{i}c", f"1.0+{i+1}c") # Highlight correct letters in blue
                else:
                    self.display_text.tag_add("error", f"1.0+{i}c", f"1.0+{i+1}c") # Highlight incorrect letters in red

        self.display_text.config(state=tk.DISABLED)  # Disable editing again

    def clear_text(self):
        # Clear the typing area and timings
        self.all_timings[self.current_text_index] = []
        self.typing_area.delete(1.0, tk.END)
        
        print(f"Cleared text {self.current_text_index + 1}")

        # Delete video and CSV files if they exist
        video_file = f'videos/video_{self.current_text_index}.mp4'
        csv_file = f'labels/video_{self.current_text_index}.csv'
        text_file = f'labels/video_{self.current_text_index}.txt'  # Define the text file path
        if os.path.exists(video_file):
            os.remove(video_file)
            print(f"Deleted {video_file}")
        if os.path.exists(csv_file):
            os.remove(csv_file)
            print(f"Deleted {csv_file}")
        if os.path.exists(text_file):  # Check if the text file exists
            os.remove(text_file)  # Delete the text file
            print(f"Deleted {text_file}")  # Print confirmation

        # Update display to remove tick mark
        self.update_text_display()

    def next_text(self):
        if self.current_text_index < len(self.texts_to_type) - 1:
            self.current_text_index += 1
            self.update_text_display()

    def previous_text(self):
        if self.current_text_index > 0:
            self.current_text_index -= 1
            self.update_text_display()

    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.display_text.config(state=tk.NORMAL)
            self.start_time = int(time.time() * 1000)  # Record the start time in milliseconds
            self.frame_count = 0  # Initialize frame count

            # Define folder paths
            videos_folder = 'videos'
            labels_folder = 'labels'
            # ground_truth_folder = 'ground_truth'  # New folder for ground truth

            # Create folders if they do not exist
            os.makedirs(videos_folder, exist_ok=True)
            os.makedirs(labels_folder, exist_ok=True)
            # os.makedirs(ground_truth_folder, exist_ok=True)  # Create ground truth folder

            # Update video filename to include the folder path
            video_filename = f'{videos_folder}/video_{self.current_text_index}.mp4'
            self.media_recorder = cv2.VideoWriter(
                video_filename,
                cv2.VideoWriter_fourcc(*'mp4v'),
                30.0,  # Assuming 30 FPS
                (640, 480)  # Assuming a resolution of 640x480
            )

            print(f"Recording started: {video_filename}")

            # Start a new thread to capture and write frames
            self.capture_thread = threading.Thread(target=self.capture_frames)
            self.capture_thread.start()

    def capture_frames(self):
        cap = cv2.VideoCapture(0)  # Open the default camera
        if not cap.isOpened():  # Check if the camera is connected
            print("Error: Camera not connected.")
            return  # Exit the function if the camera is not connected
        while self.recording:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                self.media_recorder.write(frame)  # Write the frame to the video file
                self.frame_count += 1  # Increment frame count
            else:
                break
        cap.release()

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.capture_thread.join()  # Wait for the capture thread to finish
            self.media_recorder.release()
            print("Recording stopped")

            # Calculate and print FPS
            end_time = int(time.time() * 1000)  # Get end time in milliseconds
            duration = (end_time - self.start_time) / 1000.0  # Convert duration to seconds
            fps = self.frame_count / duration if duration > 0 else 0
            print(f"Recorded FPS: {fps:.2f}")

            # Define file paths
            video_file = f'videos/video_{self.current_text_index}.mp4'  # Save video in 'videos' folder
            csv_file = f'labels/video_{self.current_text_index}.csv'      # Save CSV in 'labels' folder
            text_file = f'labels/video_{self.current_text_index}.txt'     # Save TXT in 'labels' folder

            # Save key timings to CSV
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Key', 'Frame'])  # Write header
                for key, milliseconds in self.all_timings[self.current_text_index]:
                    writer.writerow([key, milliseconds])
            print(f"Key timings saved to {csv_file}")

            # Save typed text to a text file
            with open(text_file, mode='w') as text_file:
                typed_text = self.typing_area.get(1.0, tk.END).strip()  # Get the typed text
                text_file.write(typed_text)  # Write the typed text to the file
            print(f"Typed text saved to {text_file}")

            self.update_text_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = TypingTestApp(root)
    root.mainloop()
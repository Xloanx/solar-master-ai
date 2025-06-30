# backend/app/services/watcher.py
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from loaders import build_vectorstore

DOC_DIR = "app/services/documents"

class NewDocHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith(".pdf"):
            print(f"New document detected: {event.src_path}")
            build_vectorstore()

def start_watching():
    event_handler = NewDocHandler()
    observer = Observer()
    observer.schedule(event_handler, DOC_DIR, recursive=False)
    observer.start()
    print("Watching for new documents... Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

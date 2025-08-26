import time

class Timer:
    def __init__(self):
        self.starts = {}
        self.ends = {}

    def start_time(self, tag):
        self.starts[tag] = time.time()
    
    def end_time(self, tag):
        self.ends[tag] = time.time() - self.starts[tag]
    
    def print(self):
        end2tag = {t:tag for tag, t in self.ends.items()}
        times = sorted([t for t in end2tag], reverse=True)
        print("----------------------------------------------\nTIMER\n----------------------------------------------")
        for t in times:
            tag = end2tag[t]
            print(f"{tag:<30} {round(t, 5):>10.5f}")

        for t in self.starts:
            if t in self.ends: continue
            print(f"Not Closed:\t\t{t}")
        print("----------------------------------------------")

    def reset(self):
        self.starts, self.ends = {}, {}


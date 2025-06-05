import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def select_files():
    root = tk.Tk()
    root.withdraw()
    filenames = []
    titles = ['NS file', 'EW file', 'UD file']
    for t in titles:
        f = filedialog.askopenfilename(title=f'Select {t}')
        if not f:
            messagebox.showerror('Error', 'File selection cancelled')
            return None
        filenames.append(f)
    return filenames


def ask_header(file):
    top = tk.Toplevel()
    top.title('Header line count')
    # read first 20 lines
    lines = []
    try:
        with open(file, 'r', errors='ignore') as f:
            for _ in range(20):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip())
    except Exception as e:
        lines.append(str(e))
    text = tk.Text(top, width=80, height=20)
    text.pack()
    for l in lines:
        text.insert('end', l + '\n')
    frame = tk.Frame(top)
    frame.pack(pady=5)
    tk.Label(frame, text='Header lines:').pack(side='left')
    entry = tk.Entry(frame)
    entry.insert(0, '0')
    entry.pack(side='left')

    result = {'value': None}

    def on_ok():
        try:
            result['value'] = int(entry.get())
        except ValueError:
            messagebox.showerror('Error', 'Please enter an integer')
            return
        top.destroy()

    tk.Button(top, text='OK', command=on_ok).pack(pady=5)
    top.grab_set()
    top.wait_window()
    return result['value']


def load_data(files, skiprows):
    data = []
    for f in files:
        arr = np.loadtxt(f, skiprows=skiprows)
        data.append(arr)
    return data


class WaveformGUI:
    def __init__(self, files, data, dt=0.01):
        self.files = files
        self.data = data
        self.dt = dt
        self.n_points = 8192
        self.clicks = []
        self.lines = []
        self.root = tk.Tk()
        self.root.title('Waveform FFT GUI')
        self._create_widgets()
        self._plot_waveforms()
        self.root.mainloop()

    def _create_widgets(self):
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
        self.fig = fig
        self.axes = axes
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        self.canvas.mpl_connect('button_press_event', self.on_click)

        frame = tk.Frame(self.root)
        frame.pack(fill='x')
        self.label_n = tk.Label(frame, text=f'解析データ点数: {self.n_points}')
        self.label_n.pack(side='left', padx=5)
        tk.Button(frame, text='FFT実行', command=self.run_fft).pack(side='left', padx=5)
        tk.Button(frame, text='解析データ点数×2', command=self.double_points).pack(side='left', padx=5)
        tk.Button(frame, text='解析データ点数÷2', command=self.half_points).pack(side='left', padx=5)
        tk.Button(frame, text='画面保存', command=self.save_fig).pack(side='left', padx=5)

    def _plot_waveforms(self):
        t = np.arange(len(self.data[0])) * self.dt
        labels = ['NS', 'EW', 'UD']
        for ax, d, label in zip(self.axes, self.data, labels):
            ax.clear()
            ax.plot(t, d, label=label)
            ax.set_ylabel(label)
        self.axes[-1].set_xlabel('Time [s]')
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes not in self.axes:
            return
        if len(self.clicks) == 2:
            # reset previous lines
            for l in self.lines:
                l.remove()
            self.lines.clear()
            self.clicks.clear()
        self.clicks.append(event.xdata)
        for ax in self.axes:
            line = ax.axvline(event.xdata, color='r', linestyle='--')
            self.lines.append(line)
        self.canvas.draw()

    def get_analysis_indices(self):
        if len(self.clicks) == 2:
            t0, t1 = sorted(self.clicks)
            i0 = int(t0 / self.dt)
            i1 = int(t1 / self.dt)
            i0 = max(0, i0)
            i1 = min(len(self.data[0]), i1)
        else:
            i0 = 0
            i1 = len(self.data[0])
        return i0, i1

    def double_points(self):
        self.n_points *= 2
        self.label_n.config(text=f'解析データ点数: {self.n_points}')

    def half_points(self):
        self.n_points = max(2, self.n_points // 2)
        self.label_n.config(text=f'解析データ点数: {self.n_points}')

    def save_fig(self):
        fname = filedialog.asksaveasfilename(defaultextension='.pdf', filetypes=[('PDF','*.pdf')])
        if fname:
            self.fig.savefig(fname)
            messagebox.showinfo('Saved', f'Saved to {fname}')

    def run_fft(self):
        i0, i1 = self.get_analysis_indices()
        N = self.n_points
        if i1 - i0 < N:
            messagebox.showerror('Error', 'Analysis interval shorter than block size')
            return
        shift = N // 2
        freqs = np.fft.rfftfreq(N, d=self.dt)
        specs = [np.zeros(len(freqs)) for _ in range(3)]
        count = 0
        idx = i0
        while idx + N <= i1:
            for comp, spec in zip(self.data, specs):
                seg = comp[idx:idx+N]
                spec += np.abs(np.fft.rfft(seg))
            idx += shift
            count += 1
        if count == 0:
            messagebox.showerror('Error', 'No blocks for FFT')
            return
        specs = [s / count for s in specs]
        hv = np.sqrt(specs[0]**2 + specs[1]**2) / specs[2]
        self.show_spectra(freqs, specs, hv)

    def show_spectra(self, freqs, specs, hv):
        fig, axes = plt.subplots(4, 1, figsize=(6, 8))
        labels = ['NS', 'EW', 'UD']
        for ax, spec, label in zip(axes[:3], specs, labels):
            ax.loglog(freqs, spec)
            ax.set_ylabel(label)
            ax.grid(True)
        axes[3].loglog(freqs, hv)
        axes[3].set_ylabel('H/V')
        axes[3].set_xlabel('Frequency [Hz]')
        axes[3].grid(True)
        plt.tight_layout()
        plt.show()


def main():
    files = select_files()
    if not files:
        return
    hdr = ask_header(files[0])
    if hdr is None:
        return
    data = load_data(files, hdr)
    WaveformGUI(files, data)


if __name__ == '__main__':
    main()

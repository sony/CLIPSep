"""Visualization utilities."""
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def save_loss_plot(dirname, history):
    """Plot and save the training and validation loss curves."""
    plt.figure()
    plt.plot(
        history["train"]["epoch"], history["train"]["err"], label="training",
    )
    plt.plot(
        history["val"]["epoch"], history["val"]["err"], label="validation",
    )
    plt.legend()
    plt.savefig(dirname / "loss.png", dpi=150)
    plt.close()


def save_metric_plot(dirname, history):
    """Plot and save the metric curves."""
    plt.figure()
    plt.plot(history["val"]["epoch"], history["val"]["sdr"], label="SDR")
    plt.plot(history["val"]["epoch"], history["val"]["sir"], label="SIR")
    plt.plot(history["val"]["epoch"], history["val"]["sar"], label="SAR")
    plt.legend()
    plt.savefig(dirname / "metrics.png", dpi=150)
    plt.close()


def save_loss_metric_plots(dirname, history):
    """Plot and save the loss and metric curves."""
    save_loss_plot(dirname, history)
    save_metric_plot(dirname, history)


class HTMLWriter:
    """A HTML writer for visualization."""

    def __init__(self, filename):
        self.file = open(filename, "w")
        self.file.write("<table>")
        self.file.write(
            "<style> table, th, td {border: 1px solid black;} </style>"
        )

    def write_header(self, elements):
        """Write the table header."""
        self.file.write("<tr>")
        for elem in elements:
            self.file.write(f"<th>{elem}</th>")
        self.file.write("</tr>")

    def write_elem(self, url, kind):
        """Add an element."""
        if kind == "text":
            self.file.write(url)
        elif kind == "image":
            self.file.write(
                f'<img src="{url}" style="max-height:256px;max-width:256px;">'
            )
        elif kind == "audio":
            self.file.write(f'<audio controls><source src="{url}"></audio>')
        elif kind == "video":
            self.file.write(
                f'<video src="{url}" controls="controls" '
                'style="max-height:256px;max-width:256px;">'
            )

    def write_row(self, elems):
        """Add a table row."""
        self.file.write("<tr>")
        for elem in elems:
            self.file.write("<td>")
            for kind, url in elem.items():
                self.write_elem(url, kind)
            self.file.write("</td>")
        self.file.write("</tr>")

    def write_rows(self, rows):
        """Add rows."""
        for row in rows:
            self.add_row(row)

    def close(self):
        """Close the opened file."""
        self.file.write("</table>")
        self.file.close()

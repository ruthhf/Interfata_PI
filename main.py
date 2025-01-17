from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QApplication, QPushButton, QLabel, QTabWidget, QGroupBox, QFileDialog, QTextEdit, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from collections import defaultdict

class ImageLoader(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Vizualizator imagine')
        layout = QVBoxLayout()

        # Buton pentru încărcarea imaginii
        self.load_button = QPushButton('Încarcă imagine')
        self.load_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_button)

        # Creăm un TabWidget pentru a organiza laboratoarele
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab-uri pentru fiecare laborator
        self.tabs.addTab(self.create_lab2_tab(), "Laborator 2")
        self.tabs.addTab(self.create_lab3_tab(), "Laborator 3")
        self.tabs.addTab(self.create_lab4_tab(), "Laborator 4")
        self.tabs.addTab(self.create_lab5_tab(), "Laborator 5")
        self.tabs.addTab(self.create_lab6_tab(), "Laborator 6")
        self.tabs.addTab(self.create_lab7_tab(), "Laborator 7")
        self.tabs.addTab(self.create_tema1_tab(), "Tema 1")
        self.tabs.addTab(self.create_tema2_tab(), "Tema 2")
        self.tabs.addTab(self.create_tema3_tab(), "Tema 3")
        self.tabs.addTab(self.create_tema4_tab(), "Tema 4")
        self.tabs.addTab(self.create_tema5_tab(), "Tema 5")
        
        # Label pentru afișarea imaginilor
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

    def create_lab2_tab(self):
        lab2_group = QGroupBox("Laborator 2 - Prelucrarea imaginii")
        lab2_layout = QVBoxLayout()

        self.hsv_button = QPushButton('Transformă în HSV')
        self.hsv_button.clicked.connect(self.rgb_to_hsv)
        lab2_layout.addWidget(self.hsv_button)

        self.gray_button = QPushButton('Transformă în Gri')
        self.gray_button.clicked.connect(self.rgb_to_gray)
        lab2_layout.addWidget(self.gray_button)

        self.binary_button = QPushButton('Transformă în Imagine binară')
        self.binary_button.clicked.connect(self.binary_image)
        lab2_layout.addWidget(self.binary_button)

        self.histo_gray_button = QPushButton('Histograma - Gri')
        self.histo_gray_button.clicked.connect(self.histo_gray)
        lab2_layout.addWidget(self.histo_gray_button)

        self.histo_rgb_button = QPushButton('Histograma - RGB')
        self.histo_rgb_button.clicked.connect(self.histo_rgb)
        lab2_layout.addWidget(self.histo_rgb_button)

        lab2_group.setLayout(lab2_layout)
        return lab2_group

    def create_lab3_tab(self):
        lab3_group = QGroupBox("Laborator 3 - Modificări de imagine")
        lab3_layout = QVBoxLayout()

        self.negative_button = QPushButton('Negativare imagine')
        self.negative_button.clicked.connect(self.negative_image)
        lab3_layout.addWidget(self.negative_button)

        self.contrast_button = QPushButton('Modifică Contrastul')
        self.contrast_button.clicked.connect(self.modify_contrast)
        lab3_layout.addWidget(self.contrast_button)

        self.gamma_button = QPushButton('Corecție Gamma')
        self.gamma_button.clicked.connect(self.apply_gamma_correction)
        lab3_layout.addWidget(self.gamma_button)

        self.brightness_button = QPushButton('Modifică Luminozitatea')
        self.brightness_button.clicked.connect(self.modify_brightness)
        lab3_layout.addWidget(self.brightness_button)

        lab3_group.setLayout(lab3_layout)
        return lab3_group

    def create_lab4_tab(self):
        lab4_group = QGroupBox("Laborator 4 - Filtre pe imagine")
        lab4_layout = QVBoxLayout()

        self.mean_filter_button = QPushButton('Filtru de Medie Aritmetică trece-jos')
        self.mean_filter_button.clicked.connect(self.apply_mean_filter)
        lab4_layout.addWidget(self.mean_filter_button)

        self.gaussian_filter_button = QPushButton('Filtru Gaussian trece-jos')
        self.gaussian_filter_button.clicked.connect(self.apply_gaussian_filter)
        lab4_layout.addWidget(self.gaussian_filter_button)

        self.laplacian_filter_button = QPushButton('Filtru Laplace trece-sus')
        self.laplacian_filter_button.clicked.connect(self.apply_laplacian_filter)
        lab4_layout.addWidget(self.laplacian_filter_button)

        self.custom_filter_button = QPushButton('Filtru Personalizat trece-sus')
        self.custom_filter_button.clicked.connect(self.apply_custom_filter)
        lab4_layout.addWidget(self.custom_filter_button)

        lab4_group.setLayout(lab4_layout)
        return lab4_group

    def create_lab5_tab(self):
        lab5_group = QGroupBox("Laborator 5 - Zgomot și Filtrare")
        lab5_layout = QVBoxLayout()

        self.gaussian_noise_button = QPushButton('Adaugă zgomot Gaussian')
        self.gaussian_noise_button.clicked.connect(self.add_gaussian_noise)
        lab5_layout.addWidget(self.gaussian_noise_button)

        self.bidimensional_noise_button = QPushButton('Adaugă zgomot Bidimensional')
        self.bidimensional_noise_button.clicked.connect(self.add_bidimensional_noise)
        lab5_layout.addWidget(self.bidimensional_noise_button)

        self.gaussian_filter_button = QPushButton('Filtru Gaussian pe zgomot')
        self.gaussian_filter_button.clicked.connect(self.apply_gaussian_filter_lab5)
        lab5_layout.addWidget(self.gaussian_filter_button)

        self.bidimensional_filter_button = QPushButton('Filtru Bidimensional pe zgomot')
        self.bidimensional_filter_button.clicked.connect(self.apply_bidimensional_filter_lab5)
        lab5_layout.addWidget(self.bidimensional_filter_button)

        lab5_group.setLayout(lab5_layout)
        return lab5_group

    def create_lab6_tab(self):
        lab6_group = QGroupBox("Laborator 6 - Conectivitate prin BFS")
        lab6_layout = QVBoxLayout()

        self.bfs_button = QPushButton('Etichetare prin BFS')
        self.bfs_button.clicked.connect(self.apply_bfs_labeling)
        lab6_layout.addWidget(self.bfs_button)

        self.two_pass_button = QPushButton('Etichetare prin Two-Pass')
        self.two_pass_button.clicked.connect(self.apply_two_pass_labeling) 
        lab6_layout.addWidget(self.two_pass_button)

        lab6_group.setLayout(lab6_layout)
        return lab6_group

    def create_lab7_tab(self):
        lab7_group = QGroupBox("Laborator 7")
        lab7_layout = QVBoxLayout()

        self.b_button = QPushButton('Binarizare adaptivă')
        self.b_button.clicked.connect(self.apply_binarized)
        lab7_layout.addWidget(self.b_button)

        self.dilate_button = QPushButton('Prelungirea muchiilor prin histereză')
        self.dilate_button.clicked.connect(self.apply_dilated) 
        lab7_layout.addWidget(self.dilate_button)

        lab7_group.setLayout(lab7_layout)
        return lab7_group
    
    def create_tema1_tab(self):
        tema1_group = QGroupBox("Tema 1 - Reducerea nivelelor de gri")
        tema1_layout = QVBoxLayout()

        self.threshold_button = QPushButton('Reducere nivele gri cu praguri multiple')
        self.threshold_button.clicked.connect(self.apply_threshold_reduction)
        tema1_layout.addWidget(self.threshold_button)

        self.floyd_button = QPushButton('Reducere Floyd Steinberg')
        self.floyd_button.clicked.connect(self.apply_floyd_steinberg)
        tema1_layout.addWidget(self.floyd_button)

        tema1_group.setLayout(tema1_layout)
        return tema1_group

    def create_tema2_tab(self):
        tema2_group = QGroupBox("Tema 2 - Binarizare și Egalizare")
        tema2_layout = QVBoxLayout()

        self.bin_button = QPushButton('Binarizare')
        self.bin_button.clicked.connect(self.apply_bin)
        tema2_layout.addWidget(self.bin_button)

        self.egg_button = QPushButton('Egalizare')
        self.egg_button.clicked.connect(self.apply_egg)
        tema2_layout.addWidget(self.egg_button)

        self.histo_img_button = QPushButton('Histograma')
        self.histo_img_button.clicked.connect(self.histo_img)
        tema2_layout.addWidget(self.histo_img_button)

        tema2_group.setLayout(tema2_layout)
        return tema2_group
    
    def create_tema3_tab(self):
        """Crează tab-ul pentru laboratorul 9 cu butoane pentru fiecare filtru."""
        tema3_group = QGroupBox("Tema 3 - Filtrare în domeniul frecvențial")
        tema3_layout = QVBoxLayout()

        # Butonul pentru filtrul ideal trece-jos
        self.dft_low_pass_button = QPushButton('Filtru Ideal Trece-Jos')
        self.dft_low_pass_button.clicked.connect(self.apply_dft_low_pass_ideal)
        tema3_layout.addWidget(self.dft_low_pass_button)

        # Butonul pentru filtrul ideal trece-sus
        self.dft_high_pass_button = QPushButton('Filtru Ideal Trece-Sus')
        self.dft_high_pass_button.clicked.connect(self.apply_dft_high_pass_ideal)
        tema3_layout.addWidget(self.dft_high_pass_button)

        # Butonul pentru filtrul Gaussian trece-jos
        self.dft_gaussian_low_button = QPushButton('Filtru Gaussian Trece-Jos')
        self.dft_gaussian_low_button.clicked.connect(self.apply_dft_low_pass_gaussian)
        tema3_layout.addWidget(self.dft_gaussian_low_button)

        # Butonul pentru filtrul Gaussian trece-sus
        self.dft_gaussian_high_button = QPushButton('Filtru Gaussian Trece-Sus')
        self.dft_gaussian_high_button.clicked.connect(self.apply_dft_high_pass_gaussian)
        tema3_layout.addWidget(self.dft_gaussian_high_button)

        # Eticheta pentru imagine
        self.image_label = QLabel()
        tema3_layout.addWidget(self.image_label)

        tema3_group.setLayout(tema3_layout)
        return tema3_group

    def create_tema4_tab(self):
        tema4_group = QGroupBox("Tema 4 - Extragerea conturului și umplerea regiunilor")
        tema4_layout = QVBoxLayout()

        self.extr_button = QPushButton('Contur')
        self.extr_button.clicked.connect(self.apply_extragere_contur)
        tema4_layout.addWidget(self.extr_button)

        self.umpl_button = QPushButton('Regiuni umplute')
        self.umpl_button.clicked.connect(self.apply_umplere_regiuni)
        tema4_layout.addWidget(self.umpl_button)

        tema4_group.setLayout(tema4_layout)
        return tema4_group
    
    def create_tema5_tab(self):
        tema5_group = QGroupBox("Tema 5 - Detectarea conturului")
        tema5_layout = QVBoxLayout()

        self.detect_contours_button = QPushButton("Detectare Contururi și Chain Code")
        self.detect_contours_button.clicked.connect(self.apply_detect_contours)
        tema5_layout.addWidget(self.detect_contours_button)

        # Casetă pentru afișarea codurilor
        self.chain_code_text = QTextEdit()
        self.chain_code_text.setReadOnly(True)
        tema5_layout.addWidget(self.chain_code_text)
        
        tema5_group.setLayout(tema5_layout)
        return tema5_group

    def show_image(self, img):
        if img.dtype == np.float64 or img.dtype == np.float32:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)
        if len(img.shape) == 2:  # Grayscale to RGB conversion
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:  # If the image has 3 channels (RGB/BGR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        else:
            img_rgb = img

        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(min(w, 800), min(h, 600))

    def load_image(self):
            # Încarcă imaginea utilizând un QFileDialog
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(self, "Alege imaginea", "", "Images (*.png *.xpm *.jpg *.bmp *.jpeg)", options=options)
            if file_path:
                self.current_image = cv2.imread(file_path)
                self.show_image(self.current_image)

    # Laborator 2 - Funcții pentru prelucrarea imaginii
    def rgb_to_gray(self):
        if hasattr(self, 'current_image') and self.current_image is not None:
            gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            self.show_image(gray_image)
        else:
            self.image_label.setText("Te rog să încarci o imagine mai întâi!")

    def binary_image(self):
        if hasattr(self, 'current_image') and self.current_image is not None:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            _, bw_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            self.show_image(bw_image)
        else:
            self.image_label.setText("Te rog să încarci o imagine mai întâi!")

    def rgb_to_hsv(self):
        if hasattr(self, 'current_image') and self.current_image is not None:
            hsv_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
            self.show_image(hsv_image)
        else:
            self.image_label.setText("Te rog să încarci o imagine mai întâi!")

    def histo_gray(self):
        if hasattr(self, 'current_image') and self.current_image is not None:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            plt.hist(gray.ravel(), 256, [0, 256])
            plt.title("Histograma imaginii în tonuri de gri")
            plt.xlabel("Intensitate")
            plt.ylabel("Frecvență")
            plt.show()
        else:
            self.image_label.setText("Te rog să încarci o imagine mai întâi!")

    def histo_rgb(self):
        if hasattr(self, 'current_image') and self.current_image is not None:
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                histr = cv2.calcHist([self.current_image], [i], None, [256], [0, 256])
                plt.plot(histr, color=col)
                plt.xlim([0, 256])
            plt.title("Histograma imaginii RGB")
            plt.xlabel("Intensitate")
            plt.ylabel("Frecvență")
            plt.show()
        else:
            self.image_label.setText("Te rog să încarci o imagine mai întâi!")


    # Laborator 3 - Funcții pentru modificarea imaginii
    def negative_image(self):
        if not hasattr(self, 'current_image'):
            print("Nu a fost încărcată nicio imagine.")
            return

        negative_img = 255 - self.current_image
        self.show_image(negative_img)

    def modify_contrast(self):
        if not hasattr(self, 'current_image'):
            print("Nu a fost încărcată nicio imagine.")
            return

        contrast_img = cv2.convertScaleAbs(self.current_image, alpha=2.0, beta=0)
        self.show_image(contrast_img)

    def apply_gamma_correction(self):
        if not hasattr(self, 'current_image'):
            print("Nu a fost încărcată nicio imagine.")
            return

        gamma = 3  # Poți ajusta valoarea gamma
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_img = cv2.LUT(self.current_image, table)
        self.show_image(gamma_img)

    def modify_brightness(self):
        if not hasattr(self, 'current_image'):
            print("Nu a fost încărcată nicio imagine.")
            return

        brightness_img = cv2.convertScaleAbs(self.current_image, alpha=1, beta=60)
        self.show_image(brightness_img)

    # Laborator 4 - Funcții pentru aplicarea filtrelor
    def apply_mean_filter(self):
        if not hasattr(self, 'current_image'):
            print("Nu a fost încărcată nicio imagine.")
            return

        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        kernel_mean = np.ones((3, 3), np.float32) / 9
        filtered_mean = cv2.filter2D(gray, -1, kernel_mean)
        self.show_image(filtered_mean)

    def apply_gaussian_filter(self):
        if not hasattr(self, 'current_image'):
            print("Nu a fost încărcată nicio imagine.")
            return

        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        filtered_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
        self.show_image(filtered_gaussian)

    def apply_laplacian_filter(self):
        if not hasattr(self, 'current_image'):
            print("Nu a fost încărcată nicio imagine.")
            return

        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        laplacian_filtered = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_filtered = np.uint8(np.absolute(laplacian_filtered))
        self.show_image(laplacian_filtered)

    def apply_custom_filter(self):
        if not hasattr(self, 'current_image'):
            print("Nu a fost încărcată nicio imagine.")
            return

        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        custom_kernel = np.array([[-1, -1, -1],
                                [-1, 9, -1],
                                [-1, -1, -1]])
        custom_filtered = cv2.filter2D(gray, -1, custom_kernel)
        self.show_image(custom_filtered)

    # Laborator 5 - Funcții pentru zgomot și filtrare
    def add_gaussian_noise(self):
        if not hasattr(self, 'current_image'):
            print("Nu a fost încărcată nicio imagine.")
            return

        gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        noisy_image = self.add_gaussian_noise_func(gray_image)
        self.show_image(noisy_image)

    def add_bidimensional_noise(self):
        if not hasattr(self, 'current_image'):
            print("Nu a fost încărcată nicio imagine.")
            return

        gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        noisy_image = self.add_bidimensional_noise_func(gray_image)
        self.show_image(noisy_image)

    def add_gaussian_noise_func(self, image, mean=0, std_dev=25):
        gaussian_noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
        noisy_image = image + gaussian_noise
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image.astype(np.uint8)

    def add_bidimensional_noise_func(self, image, intensity=20):
        noise = np.random.uniform(-intensity, intensity, image.shape)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image.astype(np.uint8)

    def apply_gaussian_filter_lab5(self):
        if not hasattr(self, 'current_image'):
            print("Nu a fost încărcată nicio imagine.")
            return

        gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        noisy_image = self.add_gaussian_noise_func(gray_image)
        kernel = self.gaussian_kernel(5, 1.0)
        filtered_image = self.apply_convolution(noisy_image, kernel)
        self.show_image(filtered_image)

    def apply_bidimensional_filter_lab5(self):
        if not hasattr(self, 'current_image'):
            print("Nu a fost încărcată nicio imagine.")
            return

        gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        noisy_image = self.add_bidimensional_noise_func(gray_image)
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
        kernel /= np.sum(kernel)
        filtered_image = self.apply_convolution(noisy_image, kernel)
        self.show_image(filtered_image)

    def gaussian_kernel(self, size, sigma):
        ax = np.linspace(-(size // 2), size // 2, size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / np.sum(kernel)

    def apply_convolution(self, image, kernel):
        return cv2.filter2D(image, -1, kernel)
    
    # Laborator 6
    def bfs_labeling(self, binary_image, start_i, start_j, label, labels):
            height, width = binary_image.shape
            # Direcții pentru vecinii de conectivitate de 4: sus, jos, stânga, dreapta
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            # Coada pentru BFS
            queue = deque([(start_i, start_j)])
            labels[start_i, start_j] = label

            while queue:
                i, j = queue.popleft()

                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width and binary_image[ni, nj] == 0 and labels[ni, nj] == 0:
                        labels[ni, nj] = label
                        queue.append((ni, nj))

    def bfs_connected_components(self, binary_image):
        height, width = binary_image.shape
        labels = np.zeros((height, width), dtype=int)
        current_label = 0

        for i in range(height):
            for j in range(width):
                if binary_image[i, j] == 0 and labels[i, j] == 0:
                    # Dacă pixelul nu a fost etichetat, începem un BFS de la el
                    current_label += 1
                    self.bfs_labeling(binary_image, i, j, current_label, labels)
        return labels

    def label_to_color_image(self, labels):
        max_label = labels.max()
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(max_label + 1, 3), dtype=np.uint8)
        color_image = np.ones((labels.shape[0], labels.shape[1], 3), dtype=np.uint8) * 255  # Fundal alb

        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] > 0:  # Pixel etichetat
                    color_image[i, j] = colors[labels[i, j]]

        return color_image

    def apply_bfs_labeling(self):
        if hasattr(self, 'current_image') and self.current_image is not None:
            image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            labeled_image = self.bfs_connected_components(image)
            color_image = self.label_to_color_image(labeled_image)
            self.show_image(color_image)
        else:
            self.image_label.setText("Te rog să încarci o imagine mai întâi!")
            print("No image loaded!")  # Debugging line

    #lab 6 var2 two-pass
    def two_pass_labeling(self, binary_image):
        height, width = binary_image.shape
        labels = np.zeros((height, width), dtype=int)
        current_label = 0
        equivalences = defaultdict(set)

        # Vecini (conectivitate de 4: sus, stânga)
        directions = [(-1, 0), (0, -1)]

        # Prima trecere: etichete preliminare și echivalențe
        for i in range(height):
            for j in range(width):
                if binary_image[i, j] == 0:  # Pixel de obiect (negru)
                    neighbors = []
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            if binary_image[ni, nj] == 0:  # Vecin de obiect
                                neighbors.append(labels[ni, nj])

                    if not neighbors:
                        # Niciun vecin etichetat -> etichetă nouă
                        current_label += 1
                        labels[i, j] = current_label
                    else:
                        # Atribuie cea mai mică etichetă
                        min_label = min(neighbors)
                        labels[i, j] = min_label
                        # Actualizează relațiile de echivalență
                        for neighbor_label in neighbors:
                            if neighbor_label != min_label:
                                equivalences[min_label].add(neighbor_label)
                                equivalences[neighbor_label].add(min_label)

        # Compresia claselor de echivalență (union-find)
        def find_root(label, roots):
            while roots[label] != label:
                roots[label] = roots[roots[label]]  # Compresia drumului
                label = roots[label]
            return label

        # Construiește rădăcinile etichetelor
        roots = {i: i for i in range(1, current_label + 1)}
        for label, eq_set in equivalences.items():
            for eq_label in eq_set:
                root1 = find_root(label, roots)
                root2 = find_root(eq_label, roots)
                if root1 != root2:
                    roots[root2] = root1

        # Creăm o mapare finală a etichetelor
        new_labels = {label: i + 1 for i, label in enumerate(sorted(set(find_root(label, roots) for label in roots)))}

        # A doua trecere: re-etichetează
        for i in range(height):
            for j in range(width):
                if labels[i, j] > 0:
                    labels[i, j] = new_labels[find_root(labels[i, j], roots)]

        return labels

    def label_to_color_image_two(self, labels):
        max_label = labels.max()
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(max_label + 1, 3), dtype=np.uint8)
        color_image = np.ones((labels.shape[0], labels.shape[1], 3), dtype=np.uint8) * 255  # Fundal alb

        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] > 0:  # Pixel etichetat
                    color_image[i, j] = colors[labels[i, j]]

        return color_image

    def apply_two_pass_labeling(self):
        if hasattr(self, 'current_image') and self.current_image is not None:
            image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            labeled_image = self.two_pass_labeling(image)
            color_image = self.label_to_color_image_two(labeled_image)
            self.show_image(color_image)
        else:
            self.image_label.setText("Te rog să încarci o imagine mai întâi!")

    #Laborator7
    def adaptive_threshold(self, image, block_size=11, C=2):
        height, width = image.shape
        result = np.zeros_like(image)

        half_block = block_size // 2
        for i in range(height):
            for j in range(width):
                min_i = max(0, i - half_block)
                max_i = min(height, i + half_block + 1)
                min_j = max(0, j - half_block)
                max_j = min(width, j + half_block + 1)

                window = image[min_i:max_i, min_j:max_j]
                mean_value = np.mean(window)

                result[i, j] = 255 if image[i, j] > mean_value - C else 0

        return result

    def dilate(self, image, kernel_size=3):
        height, width = image.shape
        result = np.zeros_like(image)
        half_kernel = kernel_size // 2

        for i in range(height):
            for j in range(width):
                if image[i, j] == 255:
                    for kx in range(-half_kernel, half_kernel + 1):
                        for ky in range(-half_kernel, half_kernel + 1):
                            nx, ny = i + kx, j + ky
                            if 0 <= nx < height and 0 <= ny < width:
                                result[nx, ny] = 255

        return result

    def apply_binarized(self):
        if self.current_image is None:
            print("Nu a fost încărcată nicio imagine.")
            return

        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        self.binarized = self.adaptive_threshold(gray) 
        self.show_image(self.binarized) 

    def apply_dilated(self):
        if self.binarized is None:
            print("Imaginea binarizată nu a fost generată încă. Aplică întâi binarizarea!")
            return

        self.dilated = self.dilate(self.binarized) 
        self.show_image(self.dilated)

        
    #tema1 - praguri multiple
    def rgb_to_gray_two(self):
            if self.current_image is not None:
                return cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                return None
            
    def normalize_histogram(self,image: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256], density=True)  # range [0, 256]
        return hist

    def find_histogram_peaks(self,hist: np.ndarray, window_half_width=5, threshold=0.0003) -> list:
        peaks = []
        for k in range(window_half_width, 256 - window_half_width):
            window_mean = np.mean(hist[k - window_half_width:k + window_half_width + 1])
            if hist[k] > window_mean + threshold and hist[k] >= max(hist[k - window_half_width:k + window_half_width + 1]):
                peaks.append(k)
        
        peaks.insert(0, 0)
        peaks.append(255)
        
        return peaks

    def calculate_thresholds(self,peaks: list) -> list:
        thresholds = [(peaks[i] + peaks[i + 1]) / 2 for i in range(len(peaks) - 1)]
        return thresholds

    def reduce_grayscale_levels(self,image: np.ndarray, thresholds: list, peaks: list) -> np.ndarray:
        new_image = np.zeros_like(image)
        for i in range(len(thresholds)):
            mask = (image >= thresholds[i]) & (image < (thresholds[i + 1] if i + 1 < len(thresholds) else 256))
            new_image[mask] = peaks[i]
        return new_image

    def apply_threshold_reduction(self):
        if hasattr(self, 'current_image') and self.current_image is not None:
            # Convertirea imaginii la gri
            gray_image = self.rgb_to_gray_two()

            # Normalizarea histogramei
            hist =self.normalize_histogram(gray_image)

            # Detectarea vârfurilor din histogramă
            peaks = self.find_histogram_peaks(hist)

            # Calcularea pragurilor
            thresholds = self.calculate_thresholds(peaks)

            # Reducerea nivelelor de gri
            reduced_image = self.reduce_grayscale_levels(gray_image, thresholds, peaks)

            self.reduced_image = reduced_image

            # Afișează imaginea rezultată
            self.show_image(reduced_image)
        else:
            self.image_label.setText("Te rog să încarci o imagine mai întâi!")

    #tema1 - floyd-steinberg
    def floyd_steinberg_dithering(self, image: np.ndarray) -> np.ndarray:
        dithered_image = np.zeros(image.shape, dtype=np.uint8)
        image = image.astype(np.float32)  # Convertim imaginea la float pentru a evita overflow

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                old_pixel = image[y, x]

                new_pixel = 255 if old_pixel > 127 else 0
                dithered_image[y, x] = new_pixel

                error = old_pixel - new_pixel

                if x + 1 < image.shape[1]:  
                    image[y, x + 1] += error * 7 / 16
                if y + 1 < image.shape[0]:  
                    image[y + 1, x] += error * 5 / 16
                if x - 1 >= 0 and y + 1 < image.shape[0]: 
                    image[y + 1, x - 1] += error * 3 / 16
                if x + 1 < image.shape[1] and y + 1 < image.shape[0]:  
                    image[y + 1, x + 1] += error * 1 / 16

        return dithered_image

    def apply_floyd_steinberg(self):
        if hasattr(self, 'reduced_image') and self.reduced_image is not None:
            # Aplică Floyd-Steinberg pe imaginea redusă
            dithered_image = self.floyd_steinberg_dithering(self.reduced_image)

            # Afișează imaginea rezultată
            self.show_image(dithered_image)
        else:
            self.image_label.setText("Te rog să aplici reducerea nivelelor de gri cu praguri multiple mai întâi!")

    #tema2
    def apply_bin(self):
            """Aplica binarizarea pe imaginea curentă."""
            if hasattr(self, 'current_image') and self.current_image is not None:
                imagine = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
                # Aplicăm binarizarea automată
                imagine_binarizata, prag = self.binarizare_automata(imagine)

                # Afișăm imaginea binarizată
                self.show_image(imagine_binarizata)
            else:
                print("Nu există imagine încărcată!")

    def apply_egg(self):
        """Aplica egalizarea histogramei pe imaginea curentă."""
        if hasattr(self, 'current_image') and self.current_image is not None:
            imagine=cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            # Aplicăm egalizarea histogramei
            imagine_egalizata = self.egalizare_histograma(imagine)
            self.imagine_egalizata = cv2.equalizeHist(imagine_egalizata)
            # Afișăm imaginea egalizată
            self.show_image(imagine_egalizata)
        else:
            print("Nu există imagine încărcată!")

    def histo_img(self):
        """Afișează histogramele imaginii originale și ale imaginii egalizate."""
        if hasattr(self, 'current_image') and self.current_image is not None:
            # Convertim imaginea curentă în tonuri de gri
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            
            # Calculăm histograma imaginii originale
            histo_img = cv2.calcHist([gray], [0], None, [256], [0, 256])

            # Verificăm dacă există imaginea egalizată
            if self.imagine_egalizata is not None:
                histo_egg = cv2.calcHist([self.imagine_egalizata], [0], None, [256], [0, 256])
            else:
                histo_egg = None  # Dacă imaginea nu a fost egalizată

            # Afișăm histogramele
            plt.figure(figsize=(12, 6))

            # Histograma imaginii originale
            plt.subplot(1, 2, 1)
            plt.plot(histo_img, color='blue')
            plt.title('Histograma imaginii originale')
            plt.xlabel('Intensitate')
            plt.ylabel('Număr de pixeli')

            # Histograma imaginii egalizate
            if histo_egg is not None:
                plt.subplot(1, 2, 2)
                plt.plot(histo_egg, color='green')
                plt.title('Histograma imaginii egalizate')
                plt.xlabel('Intensitate')
                plt.ylabel('Număr de pixeli')

            plt.show()

    def binarizare_automata(self, imagine, eroare=1):
        """Aplică binarizarea automată pe imaginea dată."""
        histograma, _ = np.histogram(imagine.ravel(), bins=256, range=(0, 256))

        # Găsim intensitatea max și min și inițializăm pragul
        intensitate_minima = np.min(imagine)
        intensitate_maxima = np.max(imagine)
        prag = (intensitate_minima + intensitate_maxima) / 2

        # Repetăm până când diferența între praguri este mai mică decât eroarea acceptată
        while True:
            G1 = np.sum(np.arange(0, int(prag)) * histograma[0:int(prag)]) / np.sum(histograma[0:int(prag)]) if np.sum(histograma[0:int(prag)]) > 0 else 0
            G2 = np.sum(np.arange(int(prag), 256) * histograma[int(prag):]) / np.sum(histograma[int(prag):]) if np.sum(histograma[int(prag):]) > 0 else 0

            nou_prag = (G1 + G2) / 2

            if abs(nou_prag - prag) < eroare:
                prag = nou_prag
                break

            prag = nou_prag

        # Binarizăm imaginea
        _, imagine_binarizata = cv2.threshold(imagine, prag, 255, cv2.THRESH_BINARY)
        return imagine_binarizata, prag

    def egalizare_histograma(self, imagine):
        """Aplică egalizarea histogramei pe imaginea dată."""
        # Calculăm histograma imaginii
        histograma, _ = np.histogram(imagine.ravel(), bins=256, range=(0, 256))

        # Calculăm FDPC
        fdpc = histograma.cumsum()
        fdpc_normalizat = fdpc / fdpc[-1]  # normalizăm la intervalul [0, 1]

        # Calculăm și aplicăm funcția de transformare
        transformare = (fdpc_normalizat * 255).astype(np.uint8)
        imagine_egalizata = transformare[imagine]

        return imagine_egalizata

    #tema3
    def apply_dft_low_pass_ideal(self):
        """Aplică filtrul ideal trece-jos și afișează rezultatul."""
        if self.current_image is not None:
            # Convertim imaginea în tonuri de gri
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

            # DFT și shift
            dft = np.fft.fft2(gray)
            dft_shift = np.fft.fftshift(dft)

            # Dimensiunile imaginii și centrul spectrului
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2

            # Creăm filtrul ideal trece-jos
            radius = 30
            mask_low_pass = np.zeros((rows, cols), np.uint8)
            cv2.circle(mask_low_pass, (ccol, crow), radius, 1, -1)

            # Aplicăm filtrul trece-jos
            filtered_dft_low_pass = dft_shift * mask_low_pass
            idft_shift_low_pass = np.fft.ifftshift(filtered_dft_low_pass)
            filtered_image_low_pass = np.fft.ifft2(idft_shift_low_pass)
            filtered_image_low_pass = np.abs(filtered_image_low_pass)

            # Afișăm imaginea rezultată
            self.show_image(filtered_image_low_pass)

    def apply_dft_high_pass_ideal(self):
        """Aplică filtrul ideal trece-sus și afișează rezultatul."""
        if self.current_image is not None:
            # Convertim imaginea în tonuri de gri
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

            # DFT și shift
            dft = np.fft.fft2(gray)
            dft_shift = np.fft.fftshift(dft)

            # Dimensiunile imaginii și centrul spectrului
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2

            # Creăm filtrul ideal trece-sus
            mask_low_pass = np.zeros((rows, cols), np.uint8)
            cv2.circle(mask_low_pass, (ccol, crow), 30, 1, -1)
            mask_high_pass = 1 - mask_low_pass

            # Aplicăm filtrul trece-sus
            filtered_dft_high_pass = dft_shift * mask_high_pass
            idft_shift_high_pass = np.fft.ifftshift(filtered_dft_high_pass)
            filtered_image_high_pass = np.fft.ifft2(idft_shift_high_pass)
            filtered_image_high_pass = np.abs(filtered_image_high_pass)

            # Afișăm imaginea rezultată
            self.show_image(filtered_image_high_pass)

    def apply_dft_low_pass_gaussian(self):
        """Aplică filtrul Gaussian trece-jos și afișează rezultatul."""
        if self.current_image is not None:
            # Convertim imaginea în tonuri de gri
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

            # DFT și shift
            dft = np.fft.fft2(gray)
            dft_shift = np.fft.fftshift(dft)

            # Dimensiunile imaginii și centrul spectrului
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2

            # Creăm filtrul Gaussian trece-jos
            x, y = np.meshgrid(np.arange(cols), np.arange(rows))
            gaussian_low_pass = np.exp(-((x - ccol) ** 2 + (y - crow) ** 2) / (2 * (30 ** 2)))

            # Aplicăm filtrul Gaussian trece-jos
            filtered_dft_gaussian_low = dft_shift * gaussian_low_pass
            idft_shift_gaussian_low = np.fft.ifftshift(filtered_dft_gaussian_low)
            filtered_image_gaussian_low = np.fft.ifft2(idft_shift_gaussian_low)
            filtered_image_gaussian_low = np.abs(filtered_image_gaussian_low)

            # Afișăm imaginea rezultată
            self.show_image(filtered_image_gaussian_low)

    def apply_dft_high_pass_gaussian(self):
        """Aplică filtrul Gaussian trece-sus și afișează rezultatul."""
        if self.current_image is not None:
            # Convertim imaginea în tonuri de gri
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

            # DFT și shift
            dft = np.fft.fft2(gray)
            dft_shift = np.fft.fftshift(dft)

            # Dimensiunile imaginii și centrul spectrului
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2

            # Creăm filtrul Gaussian trece-sus
            x, y = np.meshgrid(np.arange(cols), np.arange(rows))
            gaussian_low_pass = np.exp(-((x - ccol) ** 2 + (y - crow) ** 2) / (2 * (30 ** 2)))
            gaussian_high_pass = 1 - gaussian_low_pass

            # Aplicăm filtrul Gaussian trece-sus
            filtered_dft_gaussian_high = dft_shift * gaussian_high_pass
            idft_shift_gaussian_high = np.fft.ifftshift(filtered_dft_gaussian_high)
            filtered_image_gaussian_high = np.fft.ifft2(idft_shift_gaussian_high)
            filtered_image_gaussian_high = np.abs(filtered_image_gaussian_high)

            # Afișăm imaginea rezultată
            self.show_image(filtered_image_gaussian_high)

    #tema4
    def apply_extragere_contur(self):
        if hasattr(self, 'current_image') and self.current_image is not None:
            gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

            # Conversia în imagine binară
            _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

            # Extragerea conturului
            contour = self.extragere_contur(binary_image, kernel_size=3)

            # Afișarea imaginii conturului
            self.show_image(contour)

    def apply_umplere_regiuni(self):
        if hasattr(self, 'current_image') and self.current_image is not None:
            gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

            # Conversia în imagine binară
            _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

            # Seed point (se poate modifica în funcție de cerințe)
            seed_point = (50, 50)

            # Umplerea regiunilor
            filled_regions = self.umplere_regiuni(binary_image, seed_point)

            # Afișarea regiunilor umplute
            self.show_image(filled_regions)

    def extragere_contur(self, image, kernel_size=3):
        """
        Extragerea conturului folosind eroziunea imaginii.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        eroded = cv2.erode(image, kernel, iterations=1)
        contour = cv2.subtract(image, eroded)
        return contour

    def umplere_regiuni(self, image, seed_point):
        """
        Umplerea regiunilor folosind dilatare iterativă.
        """
        # Crearea imaginii pentru regiunea umplută
        filled_image = np.zeros_like(image)

        # Atribuirea punctului seed ca pixel "obiect"
        filled_image[seed_point] = 255

        # Crearea elementului structural
        kernel = np.ones((3, 3), np.uint8)

        # Dilatare iterativă
        while True:
            # Dilatarea imaginii curente
            dilated_image = cv2.dilate(filled_image, kernel, iterations=1)

            # Intersecția cu imaginea originală
            new_filled_image = cv2.bitwise_and(dilated_image, image)

            # Dacă nu există schimbări, oprirea algoritmului
            if np.array_equal(new_filled_image, filled_image):
                break

            filled_image = new_filled_image

        return filled_image

    #tema5
    def apply_detect_contours(self):
        if not hasattr(self, 'current_image') or self.current_image is None:
            QMessageBox.warning(self, "Eroare", "Nicio imagine nu este încărcată!")
            return

        # Conversia în tonuri de gri
        gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        print(f"Dimensiuni imagine gri: {gray_image.shape}")

        # Conversia în imagine binară
        _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY_INV)
        print(f"Dimensiuni imagine binară: {binary_image.shape}")
        self.show_image(binary_image)

        # Detectarea contururilor
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Număr de contururi detectate: {len(contours)}")

        if len(contours) == 0:
            QMessageBox.warning(self, "Avertisment", "Nu au fost detectate contururi în imagine!")
            return

        def extract_chain_code(contour, binary_image):
            directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
            chain_code = []
            rows, cols = binary_image.shape
            
            for i in range(len(contour)):
                point = contour[i][0]
                x, y = point[0], point[1]
                for d in range(8):
                    dx, dy = directions[d]
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < cols and 0 <= new_y < rows:
                        if binary_image[new_y, new_x] == 0:
                            chain_code.append(d)
                            break
            return chain_code

        # Calcularea codurilor înlănțuite
        chain_code_text = ""
        for idx, contour in enumerate(contours):
            print(f"Conturul {idx + 1} are {len(contour)} puncte.")
            chain_code = extract_chain_code(contour, binary_image)
            chain_code_text += f"Chain code pentru conturul {idx + 1}:\n"
            chain_code_text += ', '.join(map(str, chain_code)) + "\n\n"

        self.chain_code_text.setPlainText(chain_code_text)

        # Crearea imaginii pentru contururi
        contour_image = np.ones_like(binary_image) * 255
        cv2.drawContours(contour_image, contours, -1, 0, 2)

        self.show_image(contour_image)
  
# aplicația
app = QApplication(sys.argv)

# fereastra principală
window = ImageLoader()
window.resize(1000, 300) 
window.show()

# rulăm 
sys.exit(app.exec_())

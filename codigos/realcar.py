import cv2
import numpy as np
import os

folder = "/home/geovane/artigoRITA/base/base_simara_sem_melhoria"
folder_to_save = "/home/geovane/artigoRITA/base/base_simara_realce_log"

def realce_logaritmico(img):
	l = np.log10(img.astype(np.uint16)+1)
	return (l*105.9612).astype(np.uint8)
	
if __name__ == '__main__':

	for c in os.listdir(folder):
		
		class_path = folder + '/' + c
		
		print(class_path)
		
		# Percorrendo todos os individuos presentes na classe
		for f in os.listdir(class_path):
		
			file_name = class_path + '/' + f

			print('> Reading:', file_name)
			
			sample = open(file_name)
			
			rows, cols = sample.readline().split('\t')
			
			rows, cols = int(rows), int(cols)
			
			img = np.ndarray((cols, rows), np.uint8)
			
			# Lendo os valores
			for x in list(range(cols)):
				for y in list(range(rows)):
					value = int(sample.readline())
					
					if (value < 0):
						img[x][y] = 0
					else:
						img[x][y] = value

			img = cv2.resize(img, None, fx=1/3, fy=1/3)

			realce = realce_logaritmico(img)

			filename_to_save = folder_to_save + '/' + c + '/' + f.split('.')[0] + '.jpg'

			print(filename_to_save)

			cv2.imwrite(filename_to_save, realce)

# GERAR OS HISTOGRAMAS SEM ABORDAGENS DE DIVISÃO ESPACIAL COM AS NOVAS IMAGENS REALCADAS (PRA VER SE CHEGAM NA CASA DOS 80 DE F-SCORE)
# TESTAR TAMBEM FAZER O HISTOGRAMA MANUALMENTE, COMO EU FIZ DA VEZ QUE TESTEI ISSO

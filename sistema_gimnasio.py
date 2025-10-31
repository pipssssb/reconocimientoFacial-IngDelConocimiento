import cv2
from deepface import DeepFace
import os
from datetime import datetime

# CONFIGURACIÓN
CARPETA_MIEMBROS = 'Miembros'
ARCHIVO_REGISTRO = 'asistencia_gimnasio.csv'
UMBRAL_DISTANCIA = 0.4  # Ajusta este valor si es necesario (menor = más estricto)

# Crear carpeta de miembros si no existe
if not os.path.exists(CARPETA_MIEMBROS):
    os.makedirs(CARPETA_MIEMBROS)
    print(f"Se creó la carpeta '{CARPETA_MIEMBROS}'")

# Crear archivo de registro si no existe
if not os.path.exists(ARCHIVO_REGISTRO):
    with open(ARCHIVO_REGISTRO, 'w') as f:
        f.write('Nombre,Fecha,Hora\n')
    print(f"Se creó el archivo '{ARCHIVO_REGISTRO}'")


# FUNCIONES
def cargar_miembros():
    """Carga las fotos de los miembros registrados"""
    fotos_miembros = []
    nombres_miembros = []

    archivos = os.listdir(CARPETA_MIEMBROS)

    for archivo in archivos:
        if archivo.endswith(('.jpg', '.jpeg', '.png')):
            ruta_completa = os.path.join(CARPETA_MIEMBROS, archivo)
            fotos_miembros.append(ruta_completa)
            # Obtener nombre sin extensión (nombre_apellido)
            nombre = os.path.splitext(archivo)[0]
            nombres_miembros.append(nombre)

    return fotos_miembros, nombres_miembros


def registrar_asistencia(nombre):
    """Registra la asistencia en el archivo CSV"""
    # Leer registros existentes del día
    registros_hoy = []

    if os.path.exists(ARCHIVO_REGISTRO):
        with open(ARCHIVO_REGISTRO, 'r') as f:
            lineas = f.readlines()[1:]  # Saltar encabezado
            fecha_hoy = datetime.now().strftime('%Y-%m-%d')

            for linea in lineas:
                datos = linea.strip().split(',')
                if len(datos) >= 2 and datos[1] == fecha_hoy:
                    registros_hoy.append(datos[0])

    # Solo registrar si no asistió hoy
    if nombre not in registros_hoy:
        ahora = datetime.now()
        fecha = ahora.strftime('%Y-%m-%d')
        hora = ahora.strftime('%H:%M:%S')

        with open(ARCHIVO_REGISTRO, 'a') as f:
            f.write(f'{nombre},{fecha},{hora}\n')

        print(f"Asistencia registrada para {nombre}")
        return True
    else:
        print(f"• {nombre} ya registró su asistencia hoy")
        return False


def registrar_nuevo_miembro(frame):
    """Registra un nuevo miembro en el gimnasio"""
    print("\n" + "=" * 50)
    print("   REGISTRO DE NUEVO MIEMBRO")
    print("=" * 50)

    # Solicitar nombre
    nombre = input("\nIngresa tu nombre y apellido (ej: juan_perez): ").strip()

    if not nombre:
        print("Nombre no válido")
        return False

    # Reemplazar espacios por guiones bajos
    nombre = nombre.replace(' ', '_').lower()

    # Verificar si ya existe
    archivo_foto = os.path.join(CARPETA_MIEMBROS, f"{nombre}.jpg")
    if os.path.exists(archivo_foto):
        print(f"Ya existe un miembro con el nombre '{nombre}'")
        return False

    # Guardar la foto
    cv2.imwrite(archivo_foto, frame)
    print(f"\n¡Registro exitoso!")
    print(f"   Bienvenido al gimnasio, {nombre.replace('_', ' ').title()}!")
    print(f"   Tu foto se guardó como: {nombre}.jpg")

    return True


def capturar_y_reconocer():
    """Función principal: captura imagen y reconoce al miembro"""

    # Cargar base de datos de miembros
    fotos_miembros, nombres_miembros = cargar_miembros()

    if len(fotos_miembros) == 0:
        print("\nNo hay miembros registrados en la carpeta 'Miembros'")
        print("Los usuarios primerizos deberán registrarse")

    if len(nombres_miembros) > 0:
        print(f"\n📋 Miembros registrados: {len(nombres_miembros)}")
        print(f"   {', '.join(nombres_miembros)}\n")

    # Iniciar cámara
    print("Iniciando cámara...")
    camara = cv2.VideoCapture(0)

    if not camara.isOpened():
        print("ERROR: No se pudo acceder a la cámara")
        return

    print("Cámara lista. Presiona ESPACIO para capturar o ESC para salir")

    while True:
        # Leer frame de la cámara
        ret, frame = camara.read()

        if not ret:
            print("ERROR al capturar imagen")
            break

        # Mostrar vista previa
        cv2.putText(frame, "Presiona ESPACIO para capturar",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.imshow('Sistema de asistencia - Gimnasio', frame)

        # Esperar tecla
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == 27:  # ESC para salir
            print("Saliendo...")
            break

        elif tecla == 32:  # ESPACIO para capturar
            print("\nCapturando imagen...")

            # Guardar imagen temporal
            cv2.imwrite('temp_captura.jpg', frame)

            try:
                # Detectar cara en la captura
                caras = DeepFace.extract_faces(
                    img_path='temp_captura.jpg',
                    detector_backend='opencv',
                    enforce_detection=False
                )

                if not caras:
                    print("No se detectó ninguna cara. Intenta de nuevo.")
                    cv2.putText(frame, "No se detecto cara",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)
                    cv2.imshow('Sistema de asistencia - Gimnasio', frame)
                    cv2.waitKey(2000)
                    continue

                # Buscar coincidencia (solo si hay miembros registrados)
                mejor_coincidencia = None
                menor_distancia = float('inf')

                if len(fotos_miembros) > 0:
                    print("Buscando coincidencia...")

                    for i, foto_miembro in enumerate(fotos_miembros):
                        try:
                            resultado = DeepFace.verify(
                                img1_path='temp_captura.jpg',
                                img2_path=foto_miembro,
                                model_name='VGG-Face',
                                detector_backend='opencv',
                                enforce_detection=False
                            )

                            if resultado['verified'] and resultado['distance'] < menor_distancia:
                                menor_distancia = resultado['distance']
                                mejor_coincidencia = i

                        except Exception as e:
                            continue

                # Mostrar resultado
                cara = caras[0]['facial_area']
                x, y, w, h = cara['x'], cara['y'], cara['w'], cara['h']

                if mejor_coincidencia is not None and menor_distancia < UMBRAL_DISTANCIA:
                    # RECONOCIDO
                    nombre = nombres_miembros[mejor_coincidencia]

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(frame, f"BIENVENIDO",
                                (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 255, 0), 3)
                    cv2.putText(frame, nombre.replace('_', ' ').title(),
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)

                    print(f"\n✓ ¡BIENVENIDO {nombre.replace('_', ' ').upper()}!")
                    registrar_asistencia(nombre)

                else:
                    # NO RECONOCIDO
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.putText(frame, "NO RECONOCIDO",
                                (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 3)
                    cv2.putText(frame, "Ver consola para registrarte",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2)

                    print("\nNO RECONOCIDO")

                    # Opción de registro
                    respuesta = input("\n¿Deseas registrarte en el gimnasio? (s/n): ").strip().lower()

                    if respuesta == 's' or respuesta == 'si':
                        if registrar_nuevo_miembro(frame):
                            # Recargar lista de miembros
                            fotos_miembros, nombres_miembros = cargar_miembros()
                    else:
                        print("   Visita recepción para registrarte")

                cv2.imshow('Sistema de asistencia - Gimnasio', frame)
                cv2.waitKey(3000)  # Mostrar resultado 3 segundos

            except Exception as e:
                print(f"ERROR en el procesamiento: {e}")

            finally:
                # Limpiar archivo temporal
                if os.path.exists('temp_captura.jpg'):
                    os.remove('temp_captura.jpg')

    # Cerrar todo
    camara.release()
    cv2.destroyAllWindows()


# PROGRAMA PRINCIPAL
if __name__ == "__main__":
    print("=" * 50)
    print("   SISTEMA DE ASISTENCIA - GIMNASIO")
    print("=" * 50)

    capturar_y_reconocer()

    print("\n✓ Programa finalizado")
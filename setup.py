from setuptools import setup, find_packages


setup(
    name='bilioteca-simplex-revisado',  # Nombre del paquete
    version='0.1.0',             # Versión inicial del paquete
    author='Gustavo Lopez',          # Nombre del autor
    author_email='lopezlucasg@gmail.com.com', # Email del autor
    description='Hace el método simplex revisado', # Descripción corta
    long_description=open('README.md').read(), # Descripción larga, idealmente de un archivo README.md
    long_description_content_type='text/markdown', # Tipo de formato de la descripción larga
    url='https://github.com/Gustavo9815/Simplex_Revisado.git', # URL del repositorio del proyecto
    packages=find_packages(),    # Encuentra automáticamente todos los paquetes (directorios con __init__.py)
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',     # Versión mínima de Python requerida
    install_requires=[           # Lista de dependencias del paquete
        'requests',
        'numpy>=1.18.5',
    ],
)
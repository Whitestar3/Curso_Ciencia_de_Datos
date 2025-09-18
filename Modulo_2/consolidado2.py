# ==============================
# Sistema de Compras - Libros & Bytes
# ==============================

libros = [
    {"titulo": "Data Science con Python", "autor": "Jake VanderPlas", "precio": 25500, "stock": 5},
    {"titulo": "El Quijote", "autor": "Miguel de Cervantes", "precio": 30000, "stock": 2},
    {"titulo": "1984", "autor": "George Orwell", "precio": 18750, "stock": 10},
    {"titulo": "Ficciones", "autor": "Jorge Luis Borges", "precio": 22400, "stock": 3},
    {"titulo": "La Sombra del Viento", "autor": "Carlos Ruiz Zafón", "precio": 27800, "stock": 1}
]

# Descuentos por autor
descuentos = {
    "Jake VanderPlas": 0.15,      # 15% descuento
    "George Orwell": 0.10         # 10% descuento
}

# Totales factura
total_libros_comprados = 0
monto_total_pagado = 0
ahorro_total = 0

# Mostrar libros disponibles
def mostrar_libros_disponibles():
    print("\n--- Libros disponibles ---")
    for libro in libros:
        if libro["stock"] > 1:
            print(f"- {libro['titulo']} | Autor: {libro['autor']} | Precio: ${libro['precio']} CLP | Stock: {libro['stock']}")

# Filtrar por rango de precios
def filtrar_por_precio():
    minimo = int(input("\nIngrese precio mínimo: "))
    maximo = int(input("Ingrese precio máximo: "))
    print(f"\nLibros entre ${minimo} y ${maximo} CLP:")
    encontrado = False
    for libro in libros:
        if minimo <= libro["precio"] <= maximo:
            print(f"- {libro['titulo']} | Autor: {libro['autor']} | Precio: ${libro['precio']} CLP | Stock: {libro['stock']}")
            encontrado = True
    if not encontrado:
        print("⚠️ No hay libros en ese rango de precios.")

# Comprar libros
def comprar_libros(titulo, cantidad):
    global total_libros_comprados, monto_total_pagado, ahorro_total

    for libro in libros:
        if libro["titulo"].lower() == titulo.lower():
            if cantidad <= libro["stock"]:
                descuento = descuentos.get(libro["autor"], 0)
                precio_original = libro["precio"] * cantidad
                descuento_aplicado = precio_original * descuento
                total_compra = precio_original - descuento_aplicado

                libro["stock"] -= cantidad

                total_libros_comprados += cantidad
                monto_total_pagado += total_compra
                ahorro_total += descuento_aplicado

                print(f"\nDescuento aplicado: {descuento*100:.1f}%") if descuento > 0 else None
                print(f"Compra exitosa: {cantidad} x {libro['titulo']} - Total: ${int(total_compra)} CLP")
                return
            else:
                print("❌ No hay suficiente stock disponible.")
                return
    print("❌ Libro no encontrado.")

# Factura final
def generar_factura():
    print("\n--- FACTURA FINAL ---")
    print(f"Total de libros comprados: {total_libros_comprados}")
    print(f"Monto total pagado: ${int(monto_total_pagado)} CLP")
    print(f"Ahorro total por descuentos: ${int(ahorro_total)} CLP")
    print("¡Gracias por comprar en Libros & Bytes! 📚✨")

# Menú principal
def menu():
    while True:
        print("\n--- Sistema de Compras ---")
        print("1. Mostrar libros disponibles")
        print("2. Filtrar libros por rango de precios")
        print("3. Comprar libro")
        print("4. Finalizar compra y mostrar factura")

        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            mostrar_libros_disponibles()
        elif opcion == "2":
            filtrar_por_precio()
        elif opcion == "3":
            titulo = input("Ingrese el título del libro a comprar: ")
            cantidad = int(input("Ingrese la cantidad deseada: "))
            comprar_libros(titulo, cantidad)
        elif opcion == "4":
            generar_factura()
            break
        else:
            print("⚠️ Opción no válida, intente nuevamente.")

# Ejecutar
menu()


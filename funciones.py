from flask import Flask, request, session, render_template, redirect, url_for, send_file
import pandas as pd
from io import BytesIO
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from sqlalchemy import create_engine




from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from sqlalchemy.orm import Session
import time


# def bd_llamada(query,params_tuple):
#     load_dotenv()
#     # Create SQLAlchemy engine using pymysql
#     engine = create_engine(f"mysql+pymysql://{os.getenv('DATABASE_USER')}:{os.getenv('DATABASE_PASSWORD')}@{os.getenv('DATABASE_HOST')}/{os.getenv('DATABASE_NAME')}")
#     df = pd.read_sql_query(query, engine, params=params_tuple)
#     return df


import os
import pandas as pd
from sqlalchemy import create_engine
from mysql.connector import connect
from dotenv import load_dotenv
import numpy as np

# def bd_llamada(query,params_tuple):
#     load_dotenv()
#     engine = create_engine(f"mysql+pymysql://{os.getenv('DATABASE_USER')}:{os.getenv('DATABASE_PASSWORD')}@{os.getenv('DATABASE_HOST')}/{os.getenv('DATABASE_NAME')}")
#     df = pd.read_sql_query(query, engine, params=params_tuple)
#     return df

def bd_llamada(query, params_tuple):
    load_dotenv()

    # Convertir los tipos de datos numpy a tipos de datos nativos de Python
    params_tuple = tuple(int(param) if isinstance(param, np.integer) else param for param in params_tuple)

    # Crear la conexión a la base de datos MySQL usando mysql-connector-python a través de SQLAlchemy
    engine = create_engine(f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}")

    # Ejecutar la consulta SQL y leer los resultados en un DataFrame
    df = pd.read_sql_query(query, engine, params=params_tuple)
    
    return df


# CONSULTA PARA OBTENER LOS MÓDULOS DE TODOS LOS CLIENTES: variables dummy
def lista_clientes():
    # Ajustar la consulta SQL para utilizar el marcador de posición y pasar el id_cl como parámetro
    query_zaz = "SELECT c.id, c.nombre_comercial, c.permisos, c.cp, c.municipio, c.provincia, c.id_pais, p.name AS nombre_pais FROM clientes c INNER JOIN paises p ON c.id_pais = p.id WHERE c.activo = 1 AND c.nombre_comercial IS NOT NULL AND c.deleted_at IS NULL"
    # query_zaz = "SELECT c.id ,c.nombre_comercial, c.permisos, c.cp, c.municipio, c.provincia, c.id_pais, p.name AS nombre_pais, c.created_at AS fecha_creacion FROM clientes c INNER JOIN paises p ON c.id_pais = p.id WHERE c.activo = 1 AND c.nombre_comercial IS NOT NULL AND c.deleted_at IS NULL"
    # Llamar a la función bd_llamada con la consulta y los parámetros en una tupla
    df = bd_llamada(query_zaz, params_tuple=())
    lista_clientes_ocr = ids_clientes_ocr()

    df['kds'] = df['permisos'].apply(lambda x: 1 if '"kds": {"activo": true' in x else 0)
    df['rrhh'] = df['permisos'].apply(lambda x: 1 if '"rrhh": {"activo": true' in x else 0)
    df['amparo'] = df['permisos'].apply(lambda x: 1 if '"amparo": {"activo": true' in x else 0)
    df['cocina'] = df['permisos'].apply(lambda x: 1 if '"cocina": {"activo": true' in x else 0)
    df['tareas'] = df['permisos'].apply(lambda x: 1 if '"tareas": {"activo": true' in x else 0)
    df['basicos'] = df['permisos'].apply(lambda x: 1 if '"basicos": {"activo": true' in x else 0)
    df['compras'] = df['permisos'].apply(lambda x: 1 if '"compras": {"activo": true' in x else 0)
    df['finanzas'] = df['permisos'].apply(lambda x: 1 if '"finanzas": {"activo": true' in x else 0)
    df['almacenes'] = df['permisos'].apply(lambda x: 1 if '"almacenes": {"activo": true' in x else 0)
    df['formacion'] = df['permisos'].apply(lambda x: 1 if '"formacion": {"activo": true' in x else 0)
    df['dashboards'] = df['permisos'].apply(lambda x: 1 if '"dashboards": {"activo": true' in x else 0)
    df['documentos'] = df['permisos'].apply(lambda x: 1 if '"documentos": {"activo": true' in x else 0)
    df['ocr'] = df['id'].apply(lambda x: 1 if x in lista_clientes_ocr else 0)
    df['appcc'] = df['permisos'].apply(lambda x: 1 if '"appcc": true' in x else 0)
    df['comunicacion'] = df['permisos'].apply(lambda x: 1 if '"tareas": true' in x else 0)
    # Crear una lista con el formato deseado
    lista_formatada = [f"{row['nombre_comercial']} - {row['id']}" for _, row in df.iterrows()]
    return df, lista_formatada







def lista_paises():
    query_paises = "SELECT p.id, p.name AS nombre_pais FROM paises p"
    df_paises = bd_llamada(query_paises, params_tuple=())
    lista_paises = df_paises['nombre_pais'].unique().tolist()
    lista_paises.insert(0, "Todos los países")
    lista_paises_dict = {pais: pais for pais in lista_paises}

    return lista_paises_dict

def df_locales():
    query_locales= "SELECT COUNT(l.id) AS n, p.id AS id_pais, p.name AS nombre_pais FROM locales l INNER JOIN clientes c ON l.id_cliente = c.id INNER JOIN paises p ON c.id_pais = p.id WHERE c.activo = 1 AND c.nombre_comercial IS NOT NULL AND c.deleted_at IS NULL GROUP BY p.id"
    df_locales = bd_llamada(query_locales, params_tuple=())
    return df_locales


def df_proveedores():
    query_proveedores= "SELECT COUNT(pr.id) AS n, p.id AS id_pais, p.name AS nombre_pais FROM proveedores pr INNER JOIN clientes c ON pr.id_cliente = c.id INNER JOIN paises p ON c.id_pais = p.id WHERE c.activo = 1 AND c.nombre_comercial IS NOT NULL AND c.deleted_at IS NULL GROUP BY p.id"
    df_proveedores = bd_llamada(query_proveedores, params_tuple=())
    return df_proveedores

def df_recetas():
    query_recetas= "SELECT COUNT(r.id) AS n, p.id AS id_pais, p.name AS nombre_pais FROM recetas r INNER JOIN clientes c ON r.id_cliente = c.id INNER JOIN paises p ON c.id_pais = p.id WHERE c.activo = 1 AND c.nombre_comercial IS NOT NULL AND c.deleted_at IS NULL GROUP BY p.id"
    df_recetas = bd_llamada(query_recetas, params_tuple=())
    return df_recetas

def df_productos():
    query_productos= "SELECT COUNT(prod.id) AS n, p.id AS id_pais, p.name AS nombre_pais FROM productos prod INNER JOIN clientes c ON prod.id_cliente = c.id INNER JOIN paises p ON c.id_pais = p.id WHERE c.activo = 1 AND c.nombre_comercial IS NOT NULL AND c.deleted_at IS NULL GROUP BY p.id"
    df_productos = bd_llamada(query_productos, params_tuple=())
    return df_productos


def n_filtro_pais(df, lista_paises):
    df_filtrado_pais = df[df['nombre_pais'].isin(lista_paises)]
    num_filtrado_pais = df_filtrado_pais["n"].sum()
    return num_filtrado_pais

# ----------------

# Tenemos el problema del servidor, ARREGLAR!!!!

def obtener_codigo_postal(provincia=None, municipio=None):
    if not provincia and not municipio:
        return None

    geolocator = Nominatim(user_agent="geoapiExercises")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

    # Intenta geocodificar con los campos en el orden dado
    location = None
    if provincia and municipio:
        location = geocode(f"{municipio}, {provincia}, España")
        if not location:
            # Si no se encuentra, intenta con los campos invertidos
            location = geocode(f"{provincia}, {municipio}, España")
    elif municipio:
       location = geocode(f"{municipio}, España")
    elif provincia:
        location = geocode(f"{provincia}, España")

    if location:
        location_detail = reverse((location.latitude, location.longitude), exactly_one=True)
        if location_detail and 'postcode' in location_detail.raw['address']:
            return location_detail.raw['address']['postcode']

    return None

def actualizar_cp(row):
    if row['id_pais'] == 1 and pd.isna(row['cp']):
        row['cp'] = obtener_codigo_postal(provincia=row['provincia'], municipio=row['municipio'])
    return row

# ------------------------------------------------------------



def locales_cliente(seleccion_cliente):
    id_cl = int(seleccion_cliente.split(" - ")[-1]) # Nos quedamos con el ID del cliente de la opción del selector
    query_locales_cliente = "SELECT l.nombre FROM locales l WHERE l.id_cliente = %s AND l.activo = 1 AND l.deleted_at IS NULL AND l.nombre IS NOT NULL"
    nombres_locales = bd_llamada(query_locales_cliente, params_tuple=(id_cl,))
    lista_locales = nombres_locales['nombre'].tolist()
    return lista_locales

# def fecha_creacion_cliente(seleccion_cliente, df):
#     id_cl = int(seleccion_cliente.split(" - ")[-1]) # Nos quedamos con el ID del cliente de la opción del selector
#     filtro_fecha_creacion = df[df['id'] == id_cl]
#     if not filtro_fecha_creacion.empty:
#         fecha_creacion = filtro_fecha_creacion['fecha_creacion'].values[0]
#         fecha_creacion = pd.to_datetime(fecha_creacion).dt.date
#         return fecha_creacion
#     else:
#         fecha_str = "2020-02-27"
#         fecha_obj = datetime.strptime(fecha_str, '%Y-%m-%d').date()
#         return fecha_creacion


# ---------------------------------------------------------------------------------------------------------------------------



















# ------------------------------------------------------------------------------
# App Acordeón : módulo de RRHH

import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def obtener_datos_rrhh(cliente_id, fecha_rinicio, fecha_rfin):

    # query_rrhh1 = f"""
    # SELECT l.id_cliente, l.id, c.nombre_comercial AS nombre_cliente, l.nombre AS nombre_local, 
	# 	   (SELECT COUNT(*) FROM users u WHERE u.activo = 1 AND u.deleted_at IS NULL AND u.id_cliente = c.id) AS n_empleados,
	# 	   (SELECT COUN   T(*) FROM control_fichaje cf WHERE cf.id_local IN (SELECT l.id FROM locales l WHERE l.id_cliente = c.id AND l.activo = 1 AND l.deleted_at IS NULL)) AS n_fichajes,
	# 	   (SELECT COUNT(*) FROM vacaciones v WHERE v.id_cliente = c.id AND v.deleted_at IS NULL AND (v.aceptado = 0 OR v.aceptado = 1)) AS n_sol_vac,
	# 	   (SELECT COUNT(*) FROM vacaciones v WHERE v.id_cliente = c.id AND v.deleted_at IS NULL AND v.aceptado = 1) AS n_sol_vac_acept,
    #        (SELECT COUNT(*) FROM servicios_asignaciones sa INNER JOIN departamentos_partidas dp ON sa.id_partida = dp.id 
    #             WHERE sa.id_local IN (SELECT l2.id FROM locales l2 WHERE l2.id_cliente = c.id) AND sa.deleted_at IS NULL AND sa.visible = 1
    #             AND sa.fecha_inicio >= CURDATE() AND sa.fecha_inicio < CURDATE() + INTERVAL 1 DAY) AS n_planificaciones_hoy             
    # FROM clientes c
    # INNER JOIN locales l ON l.id_cliente = c.id 
    # WHERE c.id = %s 
    # """


    # query_rrhh1 = """SELECT c.id AS id_cliente, 
    # c.nombre_comercial AS nombre_cliente, 
    # COALESCE(SUM(u.n_empleados), 0) AS n_empleados,
    # COALESCE(SUM(cf.n_fichajes), 0) AS n_fichajes,
    # COALESCE(SUM(v.n_sol_vac), 0) AS n_sol_vac,
    # COALESCE(SUM(v_acept.n_sol_vac_acept), 0) AS n_sol_vac_acept,
    # COALESCE(SUM(sa.n_planificaciones_hoy), 0) AS n_planificaciones_hoy             
    # FROM clientes c
    # LEFT JOIN (SELECT u.id_cliente, COUNT(*) AS n_empleados FROM users u 
    #     WHERE u.activo = 1 AND u.deleted_at IS NULL GROUP BY u.id_cliente) u ON u.id_cliente = c.id
    # LEFT JOIN (SELECT l.id_cliente, COUNT(*) AS n_fichajes FROM control_fichaje cf INNER JOIN locales l ON cf.id_local = l.id
    #     WHERE l.activo = 1 AND   l.deleted_at IS NULL GROUP BY l.id_cliente) cf ON cf.id_cliente = c.id
    # LEFT JOIN (SELECT v.id_cliente, COUNT(*) AS n_sol_vac FROM vacaciones v 
    #     WHERE v.deleted_at IS NULL AND (v.aceptado = 0 OR v.aceptado = 1) GROUP BY v.id_cliente) v ON v.id_cliente = c.id
    # LEFT JOIN (SELECT v.id_cliente, COUNT(*) AS n_sol_vac_acept FROM vacaciones v 
    #     WHERE v.deleted_at IS NULL AND v.aceptado = 1 GROUP BY v.id_cliente) v_acept ON v_acept.id_cliente = c.id
    # LEFT JOIN (SELECT l2.id_cliente, COUNT(*) AS n_planificaciones_hoy FROM servicios_asignaciones sa
    #         INNER JOIN locales l2 ON sa.id_local = l2.id INNER JOIN departamentos_partidas dp ON sa.id_partida = dp.id
    #     WHERE sa.deleted_at IS NULL AND sa.visible = 1 AND sa.fecha_inicio >= CURDATE() AND sa.fecha_inicio < CURDATE() + INTERVAL 1 DAY GROUP BY l2.id_cliente) sa ON sa.id_cliente = c.id
    # WHERE c.id = %s GROUP BY c.id, c.nombre_comercial
    # """

    query_rrhh1 =f"""
    SELECT COUNT(*) AS n_empleados FROM users u
    WHERE u.activo = 1 AND u.deleted_at IS NULL AND u.id_cliente = %s
    """    

    query_rrhh2 = f"""
    SELECT r.nombre AS ROL, COUNT(*) AS N_EMPLEADOS  FROM users u INNER JOIN clientes c ON u.id_cliente = c.id
    INNER JOIN roles r ON u.id_role = r.id
    WHERE u.id IN (SELECT u.id FROM users u WHERE u.activo = 1 AND u.deleted_at IS NULL AND u.id_cliente = %s)
    GROUP BY r.id ORDER BY N_EMPLEADOS DESC
    """

    query_rrhh3 = f"""
    SELECT v.fecha_inicio, v.fecha_fin, v.aceptado FROM vacaciones v WHERE v.id_cliente = %s AND v.deleted_at IS NULL AND (v.aceptado = 1 OR v.aceptado = 0) 
    AND v.deleted_at IS NULL
    """

    query_rrhh4 = f"""SELECT COUNT(*) AS n_fichajes FROM control_fichaje cf 
    INNER JOIN locales l ON cf.id_local = l.id 
    WHERE l.activo = 1 AND l.deleted_at IS NULL AND l.id_cliente = %s
    AND cf.fecha_entrada BETWEEN %s AND %s
    """ 
    query_rrhh5 = f"""SELECT COUNT(*) AS n_sol_vac, COUNT(CASE WHEN v.aceptado = 1 THEN 1 END) AS n_sol_vac_acept
    FROM vacaciones v
    WHERE v.deleted_at IS NULL AND v.aceptado IN (0, 1, 2) AND v.id_cliente = %s 
    AND v.created_at BETWEEN %s AND %s 
    """
    query_rrhh6 = f"""SELECT COUNT(*) AS n_planificaciones_hoy FROM servicios_asignaciones sa 
    WHERE sa.deleted_at IS NULL AND sa.visible = 1 
    AND sa.fecha_inicio >= CURDATE() AND sa.fecha_inicio < CURDATE() + INTERVAL 1 DAY 
    AND sa.id_local IN (SELECT l.id FROM locales l WHERE l.id_cliente = %s AND l.deleted_at IS NULL AND l.activo = 1)
    """

    query_rrhh7 = f"""SELECT COUNT(*) AS N, vt.nombre AS tipo_ausencia FROM vacaciones v 
    INNER JOIN vacaciones_tipo vt ON v.id_tipo = vt.id 
    WHERE v.id_tipo IS NOT NULL AND v.id_cliente = %s AND v.created_at BETWEEN %s AND %s 
    GROUP BY v.id_tipo, vt.nombre ORDER BY N DESC"""


    modulo_rrhh_df1 = bd_llamada(query_rrhh1, params_tuple=(cliente_id,))
    modulo_rrhh_df2 = bd_llamada(query_rrhh2, params_tuple=(cliente_id,))
    modulo_rrhh_df3 = bd_llamada(query_rrhh3, params_tuple=(cliente_id,))
    modulo_rrhh_df4 = bd_llamada(query_rrhh4, params_tuple=(cliente_id, fecha_rinicio, fecha_rfin,))
    modulo_rrhh_df5 = bd_llamada(query_rrhh5, params_tuple=(cliente_id, fecha_rinicio, fecha_rfin,))
    modulo_rrhh_df6 = bd_llamada(query_rrhh6, params_tuple=(cliente_id,))
    modulo_rrhh_df7 = bd_llamada(query_rrhh7, params_tuple=(cliente_id, fecha_rinicio, fecha_rfin,))

    return modulo_rrhh_df1, modulo_rrhh_df2, modulo_rrhh_df3, modulo_rrhh_df4, modulo_rrhh_df5, modulo_rrhh_df6, modulo_rrhh_df7


def plot_vacation_distribution(df):
    df_copy = df.copy()

    # Convertir las columnas de fecha a datetime, manejando errores
    df_copy['fecha_inicio'] = pd.to_datetime(df_copy['fecha_inicio'], errors='coerce')
    df_copy['fecha_fin'] = pd.to_datetime(df_copy['fecha_fin'], errors='coerce')
    df_copy = df_copy.dropna(subset=['fecha_inicio', 'fecha_fin'])

    # Función para crear un rango de fechas para cada registro
    def create_date_range(row):
        return pd.date_range(start=row['fecha_inicio'], end=row['fecha_fin'], freq='D')

    # Aplicar la función para crear rangos de fechas
    try:
        df_copy['rango_fechas'] = df_copy.apply(create_date_range, axis=1)
        df_exploded = df_copy.explode('rango_fechas')

        # Filtrar las fechas para solo incluir las del año 2024
        df_exploded = df_exploded[df_exploded['rango_fechas'].dt.year == 2024]

        # Depuración: verificar el contenido de df_exploded después del filtrado
        print("Contenido de df_exploded después de filtrar por el año 2024:")
        print(df_exploded)

        # Asignar valores 'No' y 'Sí' a la columna 'aceptado'
        df_exploded['aceptado'] = df_exploded['aceptado'].replace({0: 'No', 1: 'Sí'})

        # Extraer el mes y año de cada fecha
        df_exploded['mes'] = df_exploded['rango_fechas'].dt.to_period('M')

        # Contar los días por mes y aceptación
        df_counts = df_exploded.groupby(['mes', 'aceptado']).size().unstack(fill_value=0)

        # Depuración: verificar el contenido de df_counts después del agrupamiento
        print("Contenido de df_counts después del agrupamiento:")
        print(df_counts)

        # Asegurarse de que las columnas 'Sí' y 'No' existan en el DataFrame
        if 'Sí' not in df_counts.columns:
            df_counts['Sí'] = 0
        if 'No' not in df_counts.columns:
            df_counts['No'] = 0

        df_counts = df_counts[['Sí', 'No']]

        # Verificar el contenido de df_counts antes de graficar
        print("Contenido de df_counts:")
        print(df_counts)

    except:
        print("El DataFrame df_counts está vacío después del procesamiento.")
        # Crear una figura vacía para evitar errores en el resto del código
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No hay datos para mostrar', ha='center', va='center', fontsize=12)
        ax.set_title('Distribución de Vacaciones por Mes en 2024')
        ax.set_xlabel('Mes')
        ax.set_ylabel('Días de Vacaciones')
        return fig

    # Crear el gráfico de barras apiladas
    fig, ax = plt.subplots(figsize=(10, 6))
    df_counts.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Distribución de Vacaciones por Mes en 2024')
    ax.set_xlabel('Mes')
    ax.set_ylabel('Días de Vacaciones')
    ax.legend(title='Aceptado', labels=['Sí', 'No'])
    ax.set_xticklabels(df_counts.index.strftime('%Y-%m'), rotation=45)

    return fig

# Función para convertir el gráfico matplotlib a HTML
def fig_to_html(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    html = f'<img src="data:image/png;base64,{image_base64}"/>'
    return html

def plotly_to_html(fig):
    return fig.to_html(full_html=False)










# -------------------------------------------------------------------------------------------------------

# App Acordeón : módulo de COCINA


import plotly.graph_objects as go

def obtener_datos_cocina(cliente_id):

    query_cocina1 = f"""
    SELECT c.id, 
    (SELECT COUNT(*) FROM recetas r WHERE r.id_cliente = c.id AND r.activo = 1 AND r.deleted_at IS NULL) AS n_recetas_activas,
    (SELECT COUNT(*) FROM recetas r WHERE r.id_cliente = c.id AND r.activo = 0 AND r.deleted_at IS NULL) AS n_recetas_inactivas,
    (SELECT COUNT(*) FROM productos p WHERE p.id_cliente = c.id AND p.is_fichas_tecnicas_franquicias = 1 AND p.deleted_at IS NULL
        AND p.venta = 1 AND p.id_familia_ventas IS NOT NULL AND p.activo = 1) AS n_fichas_tecnicas_activas,
    (SELECT COUNT(*) FROM productos p WHERE p.id_cliente = c.id AND p.is_fichas_tecnicas_franquicias = 1 AND p.deleted_at IS NULL 
        AND p.venta = 1 AND p.id_familia_ventas IS NOT NULL AND p.activo = 0) AS n_fichas_tecnicas_inactivas,
    (SELECT COUNT(*) FROM elaboraciones e WHERE e.id_cliente = c.id AND e.deleted_at IS NULL AND e.activo = 1) AS n_elaboraciones_activas,
    (SELECT COUNT(*) FROM elaboraciones e WHERE e.id_cliente = c.id AND e.deleted_at IS NULL AND e.activo = 0) AS n_elaboraciones_inactivas,
    (SELECT COUNT(*) FROM etiquetas e WHERE e.id_local IN (SELECT l2.id FROM locales l2 WHERE l2.id_cliente = c.id) AND e.deleted_at IS NULL AND e.favorito = 1) AS n_etiquetas_favoritas,
    (SELECT COUNT(*) FROM etiquetas e WHERE e.id_local IN (SELECT l2.id FROM locales l2 WHERE l2.id_cliente = c.id) AND e.deleted_at IS NULL AND e.favorito = 0) AS n_etiquetas_nofavoritas
FROM 
    clientes c
WHERE 
    c.id = %s
    """

    query_cocina2 = f"""
    SELECT COUNT(*) AS n_recetas_inactivas FROM recetas r WHERE r.id_cliente = %s AND r.activo = 0 AND r.deleted_at IS NULL
    """
    modulo_cocina_df1 = bd_llamada(query_cocina1, params_tuple=(cliente_id,))

    return modulo_cocina_df1 


def create_donut_chart(chart_title, center_text, colores, labels, values):
    #labels = ['ACTIVO', 'NO ACTIVO']
    #values = [active_count, inactive_count]
    #colors = ['green', 'red']  # Verde para 'ACTIVO' y rojo para 'NO ACTIVO'

    # Calcula el total para el número central
    total = sum(values)

    # Crear el gráfico de dona
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=.5, 
        marker=dict(
            colors=colores,
            line=dict(color='black', width=1)  # Bordes negros más finos
        )
    )])

    # Personaliza y agregar el número en el centro
    fig.update_layout(
        title_text=chart_title,
        annotations=[dict(text=f'{total}<br>{center_text}', x=0.5, y=0.5, font_size=15, showarrow=False, font=dict(color='black'))],
        width=500,  # Ajustar el ancho de la figura
        height=400,  # Ajustar la altura de la figura
        margin=dict(l=50, r=50, t=50, b=50),  # Ajustar los márgenes
        showlegend=True  # Mostrar la leyenda
    )

    # Devolver la figura en lugar de mostrarla directamente
    return fig



# -------------------------------------------------------------------------------------------------------

# App Acordeón : módulo de TAREAS (CHECKLISTS)

def obtener_datos_checklists(cliente_id, fecha_rinicio, fecha_rfin):

    # Consulta para obtener la información por local
    query_checklists1 = """SELECT 
        l.id_cliente,
        l.id AS id_local,
        l.nombre AS nombre_local,
        COUNT(DISTINCT c.id) AS N_TAREAS,
        COUNT(DISTINCT CASE WHEN c.activo = 1 THEN c.id END) AS N_TAREAS_ACTIVAS,
        COUNT(DISTINCT CASE WHEN c.activo = 1 AND c.cerrado = 1 THEN c.id END) AS N_TAREAS_ACTIVAS_CERRADAS,
        COUNT(DISTINCT mc.id) AS N_MODELOS,
        COUNT(DISTINCT CASE WHEN mc.activo = 1 THEN mc.id END) AS N_MODELOS_ACTIVOS,
        COUNT(DISTINCT CASE WHEN mc.activo = 1 AND mc.auditoria = 1 THEN mc.id END) AS N_MODELOS_ACTIVOS_AUDITORIAS,
        COUNT(DISTINCT CASE WHEN mc.activo = 1 AND mc.valorable = 1 THEN mc.id END) AS N_MODELOS_ACTIVOS_VALORABLES,
        COUNT(DISTINCT ac.id) AS N_ASIG,
        COUNT(DISTINCT CASE WHEN ac.activo = 1 THEN ac.id END) AS N_ASIG_ACTIVAS,
        AVG(CASE WHEN ac.activo = 1 THEN ac.prevision_dias END) AS PREVISION_DIAS_MEDIO_ACTIVAS,
        COUNT(DISTINCT CASE WHEN ac.id_periodo IS NOT NULL THEN ac.id END) AS N_RECURRENCIAS,
        COUNT(DISTINCT CASE WHEN ac.id_periodo IS NOT NULL AND ac.activo = 1 THEN ac.id END) AS N_RECURRENCIAS_ACTIVAS
    FROM 
        locales l
    LEFT JOIN 
        checklists c ON l.id = c.id_local AND c.deleted_at IS NULL AND c.fecha_inicio >= %s AND c.fecha_inicio <= %s
    LEFT JOIN 
        modelos_checklist mc ON l.id = mc.id_local AND mc.deleted_at IS NULL
    LEFT JOIN 
        asignaciones_checklist ac ON l.id = ac.id_local AND ac.deleted_at IS NULL
    WHERE l.id_cliente = %s AND l.deleted_at IS NULL AND l.activo = 1
    GROUP BY 
        l.id_cliente, l.id, l.nombre
    ORDER BY 
        l.id_cliente, l.id
    """

    if cliente_id == 1:
        query_checklists1 = """WITH Tareas AS (
                SELECT c.id_local,
                    COUNT(DISTINCT c.id) AS N_TAREAS,
                    COUNT(DISTINCT CASE WHEN c.activo = 1 THEN c.id END) AS N_TAREAS_ACTIVAS,
                    COUNT(DISTINCT CASE WHEN c.activo = 1 AND c.cerrado = 1 THEN c.id END) AS N_TAREAS_ACTIVAS_CERRADAS
                FROM checklists c
                WHERE c.deleted_at IS NULL AND c.fecha_inicio >= %s AND c.fecha_inicio <= %s
                GROUP BY c.id_local),
            Modelos AS (SELECT mc.id_local, COUNT(DISTINCT mc.id) AS N_MODELOS, COUNT(DISTINCT CASE WHEN mc.activo = 1 THEN mc.id END) AS N_MODELOS_ACTIVOS,
                    COUNT(DISTINCT CASE WHEN mc.activo = 1 AND mc.auditoria = 1 THEN mc.id END) AS N_MODELOS_ACTIVOS_AUDITORIAS,
                    COUNT(DISTINCT CASE WHEN mc.activo = 1 AND mc.valorable = 1 THEN mc.id END) AS N_MODELOS_ACTIVOS_VALORABLES
                FROM modelos_checklist mc WHERE mc.deleted_at IS NULL GROUP BY mc.id_local),
            Asignaciones AS (SELECT ac.id_local,COUNT(DISTINCT ac.id) AS N_ASIG, COUNT(DISTINCT CASE WHEN ac.activo = 1 THEN ac.id END) AS N_ASIG_ACTIVAS,
                    AVG(CASE WHEN ac.activo = 1 THEN ac.prevision_dias END) AS PREVISION_DIAS_MEDIO_ACTIVAS,
                    COUNT(DISTINCT CASE WHEN ac.id_periodo IS NOT NULL THEN ac.id END) AS N_RECURRENCIAS,
                    COUNT(DISTINCT CASE WHEN ac.id_periodo IS NOT NULL AND ac.activo = 1 THEN ac.id END) AS N_RECURRENCIAS_ACTIVAS
                FROM asignaciones_checklist ac WHERE ac.deleted_at IS NULL GROUP BY ac.id_local)
            SELECT l.id_cliente, l.id AS id_local, l.nombre AS nombre_local,
                COALESCE(t.N_TAREAS, 0) AS N_TAREAS,
                COALESCE(t.N_TAREAS_ACTIVAS, 0) AS N_TAREAS_ACTIVAS,
                COALESCE(t.N_TAREAS_ACTIVAS_CERRADAS, 0) AS N_TAREAS_ACTIVAS_CERRADAS,
                COALESCE(m.N_MODELOS, 0) AS N_MODELOS,
                COALESCE(m.N_MODELOS_ACTIVOS, 0) AS N_MODELOS_ACTIVOS,
                COALESCE(m.N_MODELOS_ACTIVOS_AUDITORIAS, 0) AS N_MODELOS_ACTIVOS_AUDITORIAS,
                COALESCE(m.N_MODELOS_ACTIVOS_VALORABLES, 0) AS N_MODELOS_ACTIVOS_VALORABLES,
                COALESCE(a.N_ASIG, 0) AS N_ASIG,
                COALESCE(a.N_ASIG_ACTIVAS, 0) AS N_ASIG_ACTIVAS,
                COALESCE(a.PREVISION_DIAS_MEDIO_ACTIVAS, 0) AS PREVISION_DIAS_MEDIO_ACTIVAS,
                COALESCE(a.N_RECURRENCIAS, 0) AS N_RECURRENCIAS,
                COALESCE(a.N_RECURRENCIAS_ACTIVAS, 0) AS N_RECURRENCIAS_ACTIVAS
            FROM locales l LEFT JOIN Tareas t ON l.id = t.id_local
            LEFT JOIN Modelos m ON l.id = m.id_local LEFT JOIN Asignaciones a ON l.id = a.id_local
            WHERE l.id_cliente = %s AND l.deleted_at IS NULL AND l.activo = 1
            ORDER BY l.id_cliente, l.id"""

    # Consulta para tabla de información por cliente 

    query_checklists2 = """SELECT
        l.id_cliente,
        COALESCE(asignaciones.N_ASIGNACIONES_ACTIVAS, 0) AS N_ASIGNACIONES_ACTIVAS,
        COALESCE(asignaciones.SUMA_PREVISION_DIAS, 0) AS SUMA_PREVISION_DIAS,
        COUNT(DISTINCT c.id) AS N_TAREAS
    FROM 
        locales l
    LEFT JOIN 
        (
            SELECT 
                l.id_cliente,
                COUNT(ac.id) AS N_ASIGNACIONES_ACTIVAS,
                SUM(ac.prevision_dias) AS SUMA_PREVISION_DIAS
            FROM 
                asignaciones_checklist ac
            JOIN 
                locales l ON ac.id_local = l.id
            WHERE 
                ac.activo = 1 
                AND ac.deleted_at IS NULL
            GROUP BY 
                l.id_cliente
        ) asignaciones ON l.id_cliente = asignaciones.id_cliente
    LEFT JOIN 
        checklists c ON l.id = c.id_local 
        AND c.deleted_at IS NULL 
        AND c.activo = 1
    WHERE 
        l.id_cliente = %s
    GROUP BY 
        l.id_cliente, asignaciones.N_ASIGNACIONES_ACTIVAS, asignaciones.SUMA_PREVISION_DIAS
    """

    query_checklists3 = """SELECT 
        COUNT(*) AS N,
        ac.id_periodo,
        ap.nombre AS nombre_periodo
    FROM 
        asignaciones_checklist ac
    INNER JOIN 
        asignaciones_periodos ap ON ac.id_periodo = ap.id
    INNER JOIN 
        locales l ON ac.id_local = l.id
    WHERE 
        l.id_cliente = %s
        AND ac.activo = 1
        AND ac.deleted_at IS NULL
    GROUP BY 
        ac.id_periodo, 
        ap.nombre
    """
    query_checklists4 = """
        SELECT prevision_dias, COUNT(*) AS N FROM asignaciones_checklist ac
        WHERE ac.id_local IN (SELECT id FROM locales l WHERE l.id_cliente = %s) 
        GROUP BY prevision_dias ORDER BY prevision_dias
        """

    modulo_checklists_df1 = bd_llamada(query_checklists1, params_tuple=(fecha_rinicio, fecha_rfin, cliente_id,))
    modulo_checklists_df1.fillna(0, inplace=True)
    modulo_checklists_df2 = bd_llamada(query_checklists2, params_tuple=(cliente_id,))
    modulo_checklists_df3 = bd_llamada(query_checklists3, params_tuple=(cliente_id,))
    modulo_checklists_df4 = bd_llamada(query_checklists4, params_tuple=(cliente_id,))

    return  modulo_checklists_df1, modulo_checklists_df2, modulo_checklists_df3, modulo_checklists_df4




def create_anillos_tareas(total_tareas, tareas_activas, tareas_cerradas):
    # Cálculo de las partes no activas y no cerradas
    tareas_no_activas = total_tareas - tareas_activas
    tareas_no_cerradas = tareas_activas - tareas_cerradas

    # Crear el gráfico de Sunburst
    fig = go.Figure()
    a = f"{total_tareas} TAREAS"
    a = "TAREAS"

    # Añadir anillo para total de tareas
    fig.add_trace(go.Sunburst(
        labels=[
            a, 
            "ACTIVAS", 
            "NO ACTIVAS", 
            "CERRADAS", 
            "NO CERRADAS"
        ],
        parents=[
            "", 
            a, 
            a, 
            "ACTIVAS", 
            "ACTIVAS"
        ],
        values=[
            total_tareas, 
            tareas_activas, 
            tareas_no_activas, 
            tareas_cerradas, 
            tareas_no_cerradas
        ],
        branchvalues="total",  # Usar los valores como partes del total del nodo padre
        hoverinfo="label+percent parent+value",  # Mostrar etiqueta, porcentaje y valor absoluto en el hover
        textinfo="label+percent parent+value",  # Mostrar etiqueta y porcentaje del padre
        insidetextorientation='radial',
        marker=dict(
            colors=[
                "#FFE680",  # Total Tareas (Amarillo pastel)
                "#FF5733",  # Tareas Activas (Naranja Fuerte)
                "#FFD1B3",  # No Activas (Naranja Pastel)
                "#4169E1",  # Tareas Cerradas (Azul Fuerte)
                "#B3D9FF"   # No Cerradas (Azul Pastel)
            ],
            line=dict(color="white", width=2)
        )
    ))

    # Configurar el diseño del gráfico
    fig.update_layout(
        title=dict(
            text="<b>Distribución de Tareas</b>",  # Usar HTML para negrita
            x=0.5,  # Centrar el título
            xanchor='center',
            font=dict(size=20, family="Arial", color="black")  # Definir fuente, tamaño y color
        ),
        margin=dict(t=50, l=50, r=50, b=50),  # Ajustar márgenes para reducir el espacio
        width=600,  # Establecer un ancho máximo para el gráfico
        height=600,  # Establecer un alto máximo para el gráfico
        showlegend=False,  # Desactivar la leyenda, no se utiliza en Sunburst
        paper_bgcolor='rgba(0,0,0,0)'  # Hacer el fondo transparente
    )

    return fig






def create_anillos_asig(total_asignaciones, total_asignaciones_activas, total_recurrencias_activas, df3_checklists):
    """
    Crea un gráfico Sunburst para mostrar la distribución de asignaciones y recurrencias, incluyendo el desglose por periodos.

    :param total_asignaciones: Número total de asignaciones
    :param total_asignaciones_activas: Número de asignaciones activas
    :param total_recurrencias_activas: Número de recurrencias activas
    :param df3_checklists: DataFrame que contiene el desglose de recurrencias por periodo
    :return: Objeto Figure de Plotly
    """

    # Calcular asignaciones no activas
    total_asignaciones_no_activas = total_asignaciones - total_asignaciones_activas
    
    # Calcular recurrencias no activas (asumimos el total de recurrencias en las activas)
    total_recurrencias_no_activas = total_asignaciones_activas - total_recurrencias_activas

    # Obtener el desglose de periodos de recurrencias activas
    periodos = df3_checklists['nombre_periodo'].tolist()
    ocurrencias = df3_checklists['N'].tolist()

    # Crear las etiquetas y los padres para el Sunburst
    labels = [
        "ASIGNACIONES",              # Nivel raíz
        "ASIG. ACT.",      # Segundo nivel
        "ASIG. NO ACT.",   # Segundo nivel
        "REC. ACT.",      # Tercer nivel sobre asignaciones activas
        "REC.S NOT AC"    # Tercer nivel sobre asignaciones activas
    ] + periodos  # Cuarto nivel sobre recurrencias activas

    parents = [
        "",                         # Raíz
        "ASIGNACIONES",             # Asignaciones activas dependen de asignaciones
        "ASIGNACIONES",             # Asignaciones no activas dependen de asignaciones
        "ASIG. ACT.",     # Recurrencias activas dependen de asignaciones activas
        "ASIG. ACT."      # Recurrencias no activas dependen de asignaciones activas
    ] + ["REC. ACT."] * len(periodos)  # Cada periodo depende de las recurrencias activas

    values = [
        total_asignaciones,                     # Total asignaciones
        total_asignaciones_activas,             # Total asignaciones activas
        total_asignaciones_no_activas,          # Asignaciones no activas
        total_recurrencias_activas,             # Recurrencias activas
        total_recurrencias_no_activas           # Recurrencias no activas
    ] + ocurrencias  # Valores de ocurrencias por periodo

    # Crear el gráfico de Sunburst
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",  # Usar los valores como partes del total del nodo padre
        hoverinfo="label+percent parent+value",  # Mostrar etiqueta, porcentaje y valor absoluto en el hover
        textinfo="label+percent parent+value",  # Mostrar etiqueta y porcentaje del padre
        insidetextorientation='radial',
        marker=dict(
            colors=[
                "#FFE680",  # Asignaciones (Amarillo pastel)
                "#FF5733",  # Asignaciones Activas (Naranja Fuerte)
                "#FFD1B3",  # Asignaciones No Activas (Naranja Pastel)
                "#32CD32",  # Recurrencias Activas (Verde Fuerte)
                "#98FB98"   # Recurrencias No Activas (Verde Pastel)
            ] + ["#C0C0C0" for _ in periodos],  # Colores para periodos
            line=dict(color="white", width=2)
        )
    ))

    # Configurar el diseño del gráfico
    fig.update_layout(
        title=dict(
            text="<b>Distribución de Asignaciones y Recurrencias</b>",  # Usar HTML para negrita
            x=0.5,  # Centrar el título
            xanchor='center',
            font=dict(size=20, family="Arial", color="black")  # Definir fuente, tamaño y color
        ),
        margin=dict(t=50, l=50, r=50, b=50),  # Ajustar márgenes para reducir el espacio
        width=800,  # Establecer un ancho máximo para el gráfico
        height=800,  # Establecer un alto máximo para el gráfico
        showlegend=False,  # Desactivar la leyenda, no se utiliza en Sunburst
        paper_bgcolor='rgba(0,0,0,0)'  # Hacer el fondo transparente
    )

    return fig


















import plotly.graph_objects as go

def crear_grafico_tareas(df, y_var, tareas_activas, tareas_cerradas, title='Tareas por Local', labels={'x': 'Cantidad de Tareas', 'y': 'Local'}):
    
    # Crear el trazo para la primera barra (tareas activas)
    barra_tareas_activas = go.Bar(
        y=df[y_var],
        x=df[tareas_activas],
        name='Tareas Activas',
        orientation='h',
        marker=dict(color='#1f77b4')
    )

    # Crear el trazo para la segunda barra (tareas cerradas)
    barra_cerradas = go.Bar(
        y=df[y_var],
        x=df[tareas_cerradas],
        name='Tareas Cerradas',
        orientation='h',
        marker=dict(color='red'),
        base=0  # Las tareas cerradas se apilan sobre las tareas activas
    )

    # Configurar el layout del gráfico
    layout = go.Layout(
        title=title,
        barmode='stack',  # Modo apilado para las barras
        xaxis=dict(title=labels['x']),
        yaxis=dict(title=labels['y']),
        height=500,  # Ajusta la altura del gráfico si es necesario
        width=600    # Ajusta el ancho del gráfico si es necesario
    )

    # Crear la figura
    fig = go.Figure(data=[barra_tareas_activas, barra_cerradas], layout=layout)

    return fig


def crear_grafico_plantillas(df, y_var, total_modelos, modelos_activos, title='Plantillas Generadas por Local', labels={'x': 'Cantidad de Plantillas', 'y': 'Local'}):
    
    # Crear el trazo para la primera barra (tareas activas
    barra_tareas_activas = go.Bar(
        y=df[y_var],
        x=df[total_modelos],
        name='Plantillas totales',
        orientation='h',
        marker=dict(color='#1f77b4')
    )

    # Crear el trazo para la segunda barra (tareas cerradas)
    barra_cerradas = go.Bar(
        y=df[y_var],
        x=df[modelos_activos],
        name='Plantillas Activas',
        orientation='h',
        marker=dict(color='green'),
        base=0 
    )

    # Configurar el layout del gráfico
    layout = go.Layout(
        title=title,
        barmode='stack',  # Modo apilado para las barras
        xaxis=dict(title=labels['x']),
        yaxis=dict(title=labels['y']),
        height=500,  # Ajusta la altura del gráfico si es necesario
        width=600    # Ajusta el ancho del gráfico si es necesario
    )

    # Crear la figura
    fig = go.Figure(data=[barra_tareas_activas, barra_cerradas], layout=layout)

    return fig









def crear_grafico_lineas(df2_checklists, title='Previsión de Días'):
    """
    Crea un gráfico de líneas para mostrar la previsión de días con una línea vertical para la media.

    :param df2_checklists: DataFrame que contiene los datos de previsión de días
    :param title: Título del gráfico
    :return: Objeto Figure de Plotly
    """
    # Asumimos que df2_checklists tiene una columna llamada 'SUMA_PREVISION_DIAS' con los datos necesarios
    df2_checklists['PREVISION_DIAS'] = df2_checklists['SUMA_PREVISION_DIAS']  # En caso de ser necesario

    # Calcular la media
    media_prevision = df2_checklists['PREVISION_DIAS'].mean()

    # Crear el gráfico de líneas
    fig = go.Figure()

    # Añadir la línea de la previsión de días
    fig.add_trace(go.Scatter(
        x=df2_checklists.index,
        y=df2_checklists['PREVISION_DIAS'],
        mode='lines+markers',
        name='Previsión de Días',
        line=dict(color='blue')
    ))

    # Añadir línea vertical para la media
    fig.add_shape(
        type='line',
        x0=media_prevision,
        y0=0,
        x1=media_prevision,
        y1=max(df2_checklists['PREVISION_DIAS']),
        line=dict(color='red', dash='dash'),
        name='Media'
    )

    # Configurar el layout del gráfico
    fig.update_layout(
        title=title,
        xaxis=dict(title='Observación'),
        yaxis=dict(title='Previsión de Días'),
        showlegend=True,
        height=400,
        width=600,
        margin=dict(t=50, l=50, r=50, b=50)
    )

    return fig


def generar_grafico_lineas(df):
    
    # Comprobar que el DataFrame tiene las columnas necesarias
    if 'prevision_dias' not in df.columns or 'N' not in df.columns:
        raise ValueError("El DataFrame debe contener las columnas 'prevision_dias' y 'N'.")
    
    # Calcular la media
    total_frecuencia = df['N'].sum()
    suma_producto = (df['prevision_dias'] * df['N']).sum()
    media = 0
    if total_frecuencia !=0:
        media = suma_producto / total_frecuencia
    
    # Crear la figura
    fig = go.Figure()

    # Añadir la traza para el gráfico de líneas
    fig.add_trace(go.Scatter(
        x=df['prevision_dias'], 
        y=df['N'], 
        mode='lines+markers',
        name='Frecuencia',
        line=dict(color='blue', width=2),
        marker=dict(color='blue', size=8)
    ))
    
    # Añadir la línea horizontal para la media
    fig.add_trace(go.Scatter(
        x=[df['prevision_dias'].min(), df['prevision_dias'].max()],
        y=[media, media],
        mode='lines',
        name=f'Media: {media:.2f}',
        line=dict(color='red', width=2, dash='dash')
    ))

    # Configurar el layout del gráfico
    fig.update_layout(
        title='Gráfico de Líneas de Frecuencia con Media',
        xaxis_title='Previsión (días)',
        yaxis_title='Frecuencia (N)',
        template='plotly',
        xaxis=dict(tickmode='linear'),
        yaxis=dict(tickmode='linear'),
        showlegend=True,
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=40, b=40),
        width=800,  # Ancho del gráfico
        height=500  # Altura del gráfico
    )

    return fig, media






# ---------------------------------------------------------------------------------

# App Acordeón : módulo de DASHBOARDS

import json

def obtener_datos_dashboards(cliente_id):
    query_dash= f"""SELECT c.permisos FROM clientes c WHERE c.id = %s"""
    modulo_dash_df = bd_llamada(query_dash, params_tuple=(cliente_id,))
    permisos_cliente = modulo_dash_df['permisos'].iloc[0]
    data = json.loads(permisos_cliente)
    dashboards_info = data.get('dashboards', {})
    return dashboards_info




# ---------------------------------------------------------------------------------

# App Acordeón : módulo de ALMACENES

def tiempo_y_frecuencia_por_local(df):
    # Convertimos por si no fuera una fecha
    df['open_at'] = pd.to_datetime(df['open_at'])
    df['closed_at'] = pd.to_datetime(df['closed_at'])
    
    # Calculamos la diferecia en segundos del tiempo en que tarda en cerrarse un inventario
    df['tiempo_en_dias'] = ((df['closed_at'] - df['open_at']).dt.total_seconds()) / (3600*24)
    
    # Filtramos filas donde closed_at no sea nulo y tiempo_en_dias sea menor a 2 días y mayor que 0
    df_filtrado = df.dropna(subset=['closed_at'])
    df_filtrado = df_filtrado[df_filtrado['tiempo_en_dias'] <= 2]
    df_filtrado = df_filtrado[df_filtrado['tiempo_en_dias'] > 0]
    
    # Agrupamos por local
    tiempo_medio = df_filtrado.groupby('id_local')['tiempo_en_dias'].mean().reset_index()
    tiempo_medio = tiempo_medio.rename(columns={'tiempo_en_dias': 'TIEMPO_MEDIO_EN_DIAS'})
    
    # Ordenamos por id_local y open_at para calcular la frecuencia de inventario
    df_sorted = df.sort_values(by=['id_local', 'open_at'])
    
    # Calculamos la diferencia entre cada open_at consecutivo por local
    df_sorted['frec'] = ( df_sorted.groupby('id_local')['open_at'].diff().dt.total_seconds() )/ (3600*24)
    
    # Calculamos el promedio de frecuencia para cada local
    frecuencia_media = df_sorted.groupby('id_local')['frec'].mean().reset_index() 
    frecuencia_media = frecuencia_media.rename(columns={'frec': 'FREC_MEDIA_EN_DIAS'})
    
    # Combinamos los resultados
    resultado = pd.merge(tiempo_medio, frecuencia_media, on='id_local')
    
    return resultado


def obtener_datos_almacen(cliente_id, fecha_rinicio, fecha_rfin):

    query_almacen1= f"""SELECT l.id, l.nombre AS LOCAL, 
        COUNT(DISTINCT au.id) AS N_ALMACENES,
        SUM((SELECT COUNT(*) 
            FROM inventarios i 
            WHERE i.open_at >= %s AND i.open_at <= %s AND i.deleted_at IS NULL AND i.id_local = l.id AND i.id_almacen_ubicacion = au.id)) AS N_INVENTARIOS,
        SUM((SELECT COUNT(*) 
            FROM inventarios i 
            WHERE i.open_at >= %s AND i.open_at <= %s AND i.deleted_at IS NULL AND i.id_local = l.id AND i.closed_at IS NULL AND i.id_almacen_ubicacion = au.id)) AS N_INVENTARIOS_ABIERTOS,
        SUM((SELECT COUNT(*) 
            FROM mermas m 
            WHERE m.fecha >= %s AND m.fecha <= %s AND m.deleted_at IS NULL AND m.id_local = l.id AND m.id_almacen_ubicacion = au.id)) AS N_MERMAS
    FROM 
        clientes c
    INNER JOIN 
        locales l ON l.id_cliente = c.id
    INNER JOIN 
        almacen_ubicacion au ON l.id = au.id_local
    WHERE 
        au.deleted_at IS NULL 
        AND au.activo = 1 
        AND c.id = %s
        AND l.deleted_at IS NULL AND l.activo = 1
    GROUP BY 
        l.id
    ORDER BY l.id"""

    query_almacen2= f"""SELECT 
        l.id, 
        SUM((SELECT COUNT(DISTINCT t.id) 
            FROM traslados t 
            inner join almacen_ubicacion au2 on au2.id = t.id_ubicacion_origen
            WHERE au2.id_local = l.id AND t.id_ubicacion_destino is not null)) AS N_TRASLADOS_ORGN,
        SUM((SELECT COUNT(DISTINCT t.id) 
            FROM traslados t 
            inner join almacen_ubicacion au2 on au2.id = t.id_ubicacion_destino
            WHERE au2.id_local = l.id AND t.id_ubicacion_origen is not null)) AS N_TRASLADOS_DTN
    FROM 
        clientes c
    INNER JOIN 
        locales l ON l.id_cliente = c.id
    WHERE 
        c.id = %s AND l.deleted_at IS NULL AND l.activo = 1
    GROUP BY 
        l.id
    ORDER BY l.id"""


    query_almacen3= f"""SELECT i.id, 
        i.id_almacen_ubicacion, 
        i.id_local,
        i.open_at, 
        i.closed_at
        FROM inventarios i 
    INNER JOIN almacen_ubicacion au ON i.id_almacen_ubicacion = au.id
    WHERE i.id_local IN (
        SELECT l.id 
        FROM locales l 
        WHERE l.id_cliente = %s 
        AND l.deleted_at IS NULL 
        AND l.activo = 1
    )
    AND i.id_almacen_ubicacion IN (
        SELECT au2.id 
        FROM almacen_ubicacion au2
        WHERE au2.deleted_at IS NULL 
        AND au2.activo = 1
    )
    AND i.open_at >= %s AND i.open_at <= %s
    """

    query_almacen4= f"""SELECT 
        l1.id AS id_local, l1.nombre AS LOCAL_ORIGEN, 
        l2.id AS id_local_destino,l2.nombre AS LOCAL_DESTINO, 
        count(DISTINCT t.id) AS N_TRASLADOS
    FROM 
        traslados t
    INNER JOIN 
        almacen_ubicacion au1 ON au1.id = t.id_ubicacion_origen 
    INNER JOIN 
        almacen_ubicacion au2 ON au2.id = t.id_ubicacion_destino 
    INNER JOIN 
        locales l1 ON au1.id_local = l1.id
    INNER JOIN 
        locales l2 ON au2.id_local = l2.id
    WHERE 
        l1.id_cliente = %s
        AND l1.deleted_at IS NULL 
        AND l1.activo = 1
        AND t.open_at >= %s AND t.open_at <= %s
    GROUP BY 
        l1.nombre, l2.nombre
    """

    query_almacen5 = f"""SELECT 
        t.id_local, 
        l_origen.nombre AS LOCAL_ORIGEN, 
        t.id_local_destino, 
        l_destino.nombre AS LOCAL_DESTINO, 
        COUNT(DISTINCT t.id) AS N_TRASLADOS
    FROM 
        traslados t
    INNER JOIN 
        locales l_origen ON t.id_local = l_origen.id
    INNER JOIN 
        locales l_destino ON t.id_local_destino = l_destino.id
    WHERE 
        t.id_local IN (SELECT l.id FROM locales l WHERE l.id_cliente = %s) 
        AND t.id_local_destino IS NOT NULL
        AND t.open_at >= %s AND t.open_at <= %s
    GROUP BY 
        t.id_local, l_origen.nombre, t.id_local_destino, l_destino.nombre
    """

    query_almacen6 = f"""SELECT
    au.id_almacen_tipo , alt.nombre, COUNT(*) AS N_ALMACEN_TIPO FROM almacen_ubicacion au 
	INNER JOIN almacen_tipo alt ON au.id_almacen_tipo = alt.id
	INNER JOIN locales l ON au.id_local = l.id 
	WHERE au.deleted_at IS NULL AND au.activo = 1 
        AND l.id_cliente = %s
        AND l.deleted_at IS NULL AND l.activo = 1
	GROUP BY au.id_almacen_tipo 
    """

    # DF para la primera consulta
    modulo_almacen_df1 = bd_llamada(query_almacen1, params_tuple=(fecha_rinicio, fecha_rfin, fecha_rinicio, fecha_rfin, fecha_rinicio, fecha_rfin, cliente_id,))
    # DF para la segunda consulta
    # modulo_almacen_df2 = bd_llamada(query_almacen2, params_tuple=(cliente_id,))
    # DF para la tercera consulta
    modulo_almacen_df3 = bd_llamada(query_almacen3, params_tuple=(cliente_id,fecha_rinicio, fecha_rfin))
    modulo_almacen_df3 = tiempo_y_frecuencia_por_local(modulo_almacen_df3)
    # Combinamos todos los df
    #df_comb1 = pd.merge(modulo_almacen_df1, modulo_almacen_df2, on='id', how='outer')
    # Combinar df_comb1 con df_3 por id_local
    df_comb_final = pd.merge(modulo_almacen_df1, modulo_almacen_df3, left_on='id', right_on='id_local', how='outer').drop(labels='id_local', axis = 1)
    # Rellenar todos los valores nulos con 0
    df_final_almacen = df_comb_final.fillna(0)
    # DataFrame que cuenta número de traslados entre locales
    df_traslados = bd_llamada(query_almacen4, params_tuple=(cliente_id,fecha_rinicio, fecha_rfin))
    if df_traslados.empty:
        # Ejecutamos esta en caso de que esté vacía
        df_traslados = bd_llamada(query_almacen5, params_tuple=(cliente_id,fecha_rinicio, fecha_rfin))

    df_tipos_almacen = bd_llamada(query_almacen6, params_tuple=(cliente_id,))

    return df_final_almacen, df_traslados, df_tipos_almacen

def contar_traslados(df):
    # Contamos traslados entre mismos y diferentes locales
    traslados_mismos_locales = df[df['id_local'] == df['id_local_destino']]['N_TRASLADOS'].sum()
    traslados_diferentes_locales = df[df['id_local'] != df['id_local_destino']]['N_TRASLADOS'].sum()  
    return traslados_mismos_locales, traslados_diferentes_locales


def heatmap_almacen(df):
    pivot_table = df.pivot_table(index="LOCAL_ORIGEN", columns="LOCAL_DESTINO", values="N_TRASLADOS", aggfunc='sum', fill_value=0)

    # Crear la escala de colores personalizada con umbrales ajustados y colores más fuertes
    colors = [
        [0.0, 'rgb(255, 255, 255)'],
        [0.02, 'rgb(255, 255, 200)'],
        [0.04, 'rgb(255, 255, 150)'],
        [0.06, 'rgb(255, 255, 100)'],
        [0.08, 'rgb(255, 255, 50)'],
        [0.1, 'rgb(255, 255, 0)'],
        [0.2, 'rgb(255, 204, 0)'],
        [0.3, 'rgb(255, 153, 0)'],
        [0.4, 'rgb(255, 102, 0)'],
        [0.5, 'rgb(255, 0, 0)'],
        [0.6, 'rgb(204, 0, 0)'],
        [0.7, 'rgb(153, 0, 0)'],
        [0.8, 'rgb(102, 0, 0)'],
        [1.0, 'rgb(0, 0, 0)']
    ]

    # Crear el mapa de calor con Plotly
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale=colors,
        zmin=0,
        showscale=True,
        colorbar=dict(
            title='Número de Traslados',
            titleside='right',
            titlefont=dict(
                family='Arial Black',
                size=14,
                color='black'
            ),
            tickfont=dict(
                family='Arial Black',
                size=12,
                color='black'
            )
        ),
        hovertemplate='ORIGEN: %{y}<br>DESTINO: %{x}<br>TRASLADOS: %{z}<extra></extra>'
    ))

    # Añadir bordes negros a cada celda
    fig.update_traces(
        zmin=0,
        zmax=pivot_table.values.max(),
        showscale=True,
        hoverongaps=False,
        selector=dict(type='heatmap')
    )

    # Añadir anotaciones para los valores en cada celda, omitiendo los ceros
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            value = pivot_table.values[i, j]
            if value != 0:  # Omitir valores cero
                fig.add_annotation(
                    dict(
                        showarrow=False,
                        text=str(value),
                        x=pivot_table.columns[j],
                        y=pivot_table.index[i],
                        font=dict(
                            color="black" if value < (pivot_table.values.max() / 2) else "white",
                            size=12,  # Incrementar el tamaño de letra
                            family="Arial Black"
                        )
                    )
                )

    # Configurar el título, los márgenes y el tamaño del gráfico
    fig.update_layout(
        title=dict(
            text='TRASLADOS ENTRE LOCALES',
            font=dict(
                family="Arial Black",
                size=24,  # Tamaño de la fuente del título
                color="black"
            )
        ),
        xaxis_title=dict(
            text='Local de Destino',
            font=dict(
                family="Arial Black",
                size=20,
                color="black"
            )
        ),
        yaxis_title=dict(
            text='Local de Origen',
            font=dict(
                family="Arial Black",
                size=20,
                color="black"
            )
        ),
        width=1200,
        height=900,
        margin=dict(l=100, r=100, t=100, b=100),  # Ajustar los márgenes
        xaxis=dict(
            tickangle=45,
            tickfont=dict(
                family="Arial Black",
                size=12,
                color="black"
            ),
            automargin=True  # Ajustar automáticamente los márgenes para las etiquetas del eje x
        ),
        yaxis=dict(
            tickfont=dict(
                family="Arial Black",
                size=12,
                color="black"
            ),
            automargin=True  # Ajustar automáticamente los márgenes para las etiquetas del eje y
        )
    )

    # Mostrar el gráfico
    return fig


def crear_grafico_sectores_almacenes(df, labels_column, values_column, titulo, colores=None):
    labels = df[labels_column]
    values = df[values_column]
    
    # Configurar colores vivos si no se proporcionan
    if colores is None:
        colores = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    
    # Crear el gráfico de sectores
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(
            colors=colores,  # Colores vivos
            line=dict(color='black', width=2)  # Bordes negros más gruesos
        )
    )])

    # Añadir el título y ajustar el layout (sin anotación central)
    fig.update_layout(
        title_text=titulo,
        width=500,  # Ajustar el ancho de la figura
        height=400,
    )
    return fig









# ---------------------------------------------------------------------------------

# App Acordeón : módulo de COMPRAS


def obtener_datos_compras(cliente_id):
    query_compras = """
    SELECT l.id_cliente, l.id, c.nombre_comercial AS nombre_cliente, l.nombre AS nombre_local
    FROM clientes c
    INNER JOIN locales l ON l.id_cliente = c.id
    WHERE c.id = %s AND l.deleted_at IS NULL AND l.activo = 1
    """
    modulo_compras_df = bd_llamada(query_compras, params_tuple=(cliente_id,))

    return modulo_compras_df


def obtener_datos_compras_2(cliente_id, fecha_rinicio, fecha_rfin):
    query_compras1 = f"""SELECT p.id_cliente, COUNT(*) AS total_productos_compra,
    SUM(CASE WHEN p.activo = 1 THEN 1 ELSE 0 END) AS productos_activos_compra,
    SUM(CASE WHEN p.id NOT IN (SELECT ptp.id_producto FROM productos_to_proveedores ptp WHERE ptp.deleted_at IS NULL) THEN 1 ELSE 0 END) AS productos_sin_proveedor
    FROM productos p
    WHERE p.deleted_at IS NULL AND p.compra = 1 AND p.id_cliente = %s
    GROUP BY p.id_cliente
    """

    query_compras2 = f"""SELECT COUNT(*) AS N, fpm.nombre AS NOMBRE_FAMILIA FROM productos p
    LEFT JOIN productos_familias pf ON p.id_familia = pf.id 
    LEFT JOIN familias_productos_master fpm ON pf.id_familia_master = fpm.id
    WHERE p.deleted_at IS NULL AND p.activo = 1 AND p.id_cliente = %s
    GROUP BY pf.id_familia_master
    """

    query_compras3 = f"""SELECT c.id AS id_cliente, l.id AS id_local, c.nombre_comercial AS nombre_cliente, l.nombre AS nombre_local,
    (SELECT COUNT(*) FROM albaranes a WHERE a.id_local = l.id AND a.deleted_at IS NULL) AS total_albaranes_generados,
    (SELECT COUNT(*) FROM albaranes a WHERE a.id_local = l.id AND a.cerrado = 0 AND a.deleted_at IS NULL) AS albaranes_por_cerrar,
    (SELECT COUNT(*) FROM albaranes a WHERE a.id_local = l.id AND a.deleted_at IS NULL 
    	AND a.id IN (SELECT DISTINCT ai.id_albaran FROM albaranes_imagnes ai WHERE ai.deleted_at IS NULL )) AS albaranes_por_foto,
    (SELECT COUNT(*) FROM pedidos p WHERE p.id_local = l.id AND p.deleted_at IS NULL) AS pedidos_generados,
    (SELECT COUNT(*) FROM facturas f WHERE f.id_local = l.id AND f.deleted_at IS NULL) AS facturas,
    (SELECT COUNT(*) FROM facturas f WHERE f.id_cliente = l.id_cliente AND f.deleted_at IS NULL AND f.id_local IS NULL) AS facturas_cliente_sin_local,
    (SELECT COUNT(*) FROM facturas f WHERE f.id_local = l.id AND f.deleted_at IS NULL AND f.id_estado = 1) AS facturas_pendiente_pago,
    (SELECT COUNT(*) FROM facturas f WHERE f.id_local = l.id AND f.deleted_at IS NULL AND f.id_estado = 2) AS facturas_pagadas,
    (SELECT COUNT(*) FROM facturas f WHERE f.id_local = l.id AND f.deleted_at IS NULL AND f.id_estado IS NULL) AS facturas_sin_estado,
    (SELECT COUNT(*) FROM proveedores pr WHERE pr.id_cliente = l.id_cliente AND pr.deleted_at IS NULL) AS total_proveedores,
    (SELECT COUNT(*) FROM proveedores pr WHERE pr.id_cliente = l.id_cliente AND pr.deleted_at IS NULL AND pr.activo = 1) AS total_proveedores_activos
    FROM clientes c 
    INNER JOIN locales l ON l.id_cliente = c.id
    WHERE c.id = %s AND l.deleted_at IS NULL AND l.activo = 1
    """

    query_compras3 = f"""SELECT c.id AS id_cliente, l.id AS id_local, c.nombre_comercial AS nombre_cliente, l.nombre AS nombre_local,
    (SELECT COUNT(*) FROM albaranes a WHERE a.id_local = l.id AND a.deleted_at IS NULL AND a.created_at >= %s AND a.created_at <= %s) AS total_albaranes_generados,
    (SELECT COUNT(*) FROM albaranes a WHERE a.id_local = l.id AND a.cerrado = 0 AND a.deleted_at IS NULL AND a.created_at >= %s AND a.created_at <= %s) AS albaranes_por_cerrar,
    (SELECT COUNT(*) FROM albaranes a WHERE a.id_local = l.id AND a.deleted_at IS NULL AND a.created_at >= %s AND a.created_at <= %s
    	AND a.id IN (SELECT DISTINCT ai.id_albaran FROM albaranes_imagnes ai WHERE ai.deleted_at IS NULL )) AS albaranes_por_foto,
    (SELECT COUNT(*) FROM pedidos p WHERE p.id_local = l.id AND p.deleted_at IS NULL AND p.created_at >= %s AND p.created_at <= %s ) AS pedidos_generados,
    (SELECT COUNT(*) FROM facturas f WHERE f.id_local = l.id AND f.deleted_at IS NULL AND f.fecha >= %s AND f.fecha <= %s) AS facturas,
    (SELECT COUNT(*) FROM facturas f WHERE f.id_cliente = l.id_cliente AND f.deleted_at IS NULL AND f.id_local IS NULL AND f.fecha >= %s AND f.fecha <= %s) AS facturas_cliente_sin_local,
    (SELECT COUNT(*) FROM facturas f WHERE f.id_local = l.id AND f.deleted_at IS NULL AND f.id_estado = 1 AND f.fecha >= %s AND f.fecha <= %s) AS facturas_pendiente_pago,
    (SELECT COUNT(*) FROM facturas f WHERE f.id_local = l.id AND f.deleted_at IS NULL AND f.id_estado = 2 AND f.fecha >= %s AND f.fecha <= %s) AS facturas_pagadas,
    (SELECT COUNT(*) FROM facturas f WHERE f.id_local = l.id AND f.deleted_at IS NULL AND f.id_estado IS NULL AND f.fecha >= %s AND f.fecha <= %s) AS facturas_sin_estado,
    (SELECT COUNT(*) FROM proveedores pr WHERE pr.id_cliente = l.id_cliente AND pr.deleted_at IS NULL) AS total_proveedores,
    (SELECT COUNT(*) FROM proveedores pr WHERE pr.id_cliente = l.id_cliente AND pr.deleted_at IS NULL AND pr.activo = 1) AS total_proveedores_activos
    FROM clientes c 
    INNER JOIN locales l ON l.id_cliente = c.id
    WHERE c.id = %s AND l.deleted_at IS NULL AND l.activo = 1"""

    query_compras4 = f"""SELECT AVG(a.importe) AS importe_medio_albaran
    FROM albaranes a WHERE a.deleted_at IS NULL AND a.cerrado = 1 
    AND a.created_at >= %s AND a.created_at <= %s
    AND a.id_local IN (SELECT l.id FROM locales l WHERE l.id_cliente = %s AND l.deleted_at IS NULL AND l.activo = 1) 
    """

    query_compras5 = f"""SELECT p.id_estado, ep.nombre_cliente AS ESTADO, COUNT(*) AS N FROM pedidos p 
    INNER JOIN estados_pedidos ep ON p.id_estado = ep.id 
    WHERE p.id_local IN ((SELECT l.id FROM locales l WHERE l.id_cliente = %s AND l.deleted_at IS NULL AND l.activo = 1))
    AND p.deleted_at IS NULL AND p.created_at >= %s AND p.created_at <= %s 
    GROUP BY ep.nombre_cliente ORDER BY N DESC
    """

    query_compras6 = f"""SELECT p.id_proveedor, p2.nombre_comercial , COUNT(*) AS N_PEDIDOS, AVG(p.importe) AS IMP_MEDIO_PEDIDO, SUM(importe) AS S_PEDIDOS FROM pedidos p
    INNER JOIN proveedores p2 ON p.id_proveedor = p2.id
    WHERE p.id_local IN (SELECT l.id FROM locales l WHERE l.id_cliente = %s AND l.deleted_at IS NULL AND l.activo = 1)
    AND p.created_at >= %s AND p.created_at <= %s
    AND p.deleted_at IS NULL GROUP BY p.id_proveedor ORDER BY p.id_proveedor ASC
    """

    query_compras7 = f"""SELECT COUNT(*) AS N_PEDIDOS, p.id_proveedor, p2.nombre_comercial, AVG(DATEDIFF(p.fecha_recepcion, p.send_at)) AS tiempo_promedio_dias
    FROM pedidos p 
    INNER JOIN proveedores p2 ON p.id_proveedor = p2.id
    WHERE p.id_local IN (SELECT l.id FROM locales l WHERE l.id_cliente = %s AND l.deleted_at IS NULL AND l.activo = 1)
        AND p.deleted_at IS NULL AND p.fecha_recepcion IS NOT NULL
        AND p.send_at IS NOT NULL 
        AND p.id_estado = 5
        AND p.created_at >= %s AND p.created_at <= %s
        AND DATE(p.send_at) <= p.fecha_recepcion
    GROUP BY 
        p.id_proveedor 
    ORDER BY 
        p.id_proveedor ASC
    """
    
    time0 = time.time()
    modulo_compras_df1 = bd_llamada(query_compras1, params_tuple=(cliente_id,))
    time1 = time.time()
    print("1--- ", time1-time0 )
    modulo_compras_df2 = bd_llamada(query_compras2, params_tuple=(cliente_id,))
    modulo_compras_df2['NOMBRE_FAMILIA'] = modulo_compras_df2['NOMBRE_FAMILIA'].replace('', 'Otros')
    modulo_compras_df2 = modulo_compras_df2.groupby('NOMBRE_FAMILIA', as_index=False)['N'].sum()
    time2 = time.time()
    print("2--- ", time2-time1)
    
    params_tuple_3 = (fecha_rinicio, fecha_rfin, fecha_rinicio, fecha_rfin, fecha_rinicio, fecha_rfin, fecha_rinicio, fecha_rfin, fecha_rinicio, fecha_rfin, fecha_rinicio, fecha_rfin, fecha_rinicio, fecha_rfin, fecha_rinicio, fecha_rfin, fecha_rinicio, fecha_rfin, cliente_id,)
    modulo_compras_df3 = bd_llamada(query_compras3, params_tuple = params_tuple_3)
    time3 = time.time()
    print("3--- ", time3 - time2)
    # modulo_compras_df4 no se está utilizando (de momento???)
    # modulo_compras_df4 = bd_llamada(query_compras4, params_tuple=(fecha_rinicio, fecha_rfin,cliente_id,))
    
    modulo_compras_df5 = bd_llamada(query_compras5, params_tuple=(cliente_id,fecha_rinicio, fecha_rfin,))
    time5 = time.time()
    print("5--- ", time5-time3)
    
    modulo_compras_df6 = bd_llamada(query_compras6, params_tuple=(cliente_id,fecha_rinicio, fecha_rfin,))
    time6 = time.time()
    print("6--- ", time6-time5)
    
    modulo_compras_df7 = bd_llamada(query_compras7, params_tuple=(cliente_id,fecha_rinicio, fecha_rfin,))
    time7 = time.time()
    print("7--- ", time7 - time6)
    modulo_compras_df7 = modulo_compras_df7.drop(columns=['N_PEDIDOS','nombre_comercial'])
    modulo_compras_df67 = pd.merge(modulo_compras_df6, modulo_compras_df7, on='id_proveedor', how='left')
    # acordarse de que se llama _2 !!!!!!!!!!
    return modulo_compras_df1, modulo_compras_df2, modulo_compras_df3, modulo_compras_df5, modulo_compras_df67



def plot_barras_compras(df, label_column, value_columns, colors, legend_names, graph_title, xaxis_title, yaxis_title, total_accumulated=True):
    """
    Genera un gráfico de barras horizontales apiladas o superpuestas, según la configuración,
    utilizando Plotly, con la leyenda fuera del área de las barras y sin fondo.

    Parámetros:
    - df (DataFrame): DataFrame que contiene los datos.
    - label_column (str): Nombre de la columna en df que contiene las etiquetas de las barras (nombres de los locales).
    - value_columns (list): Lista de nombres de columnas que contienen los valores a representar.
    - colors (list): Lista de colores para cada conjunto de barras, en el mismo orden que `value_columns`.
    - legend_names (list): Nombres para cada conjunto de barras, utilizados en la leyenda.
    - graph_title (str): Título del gráfico.
    - xaxis_title (str): Título para el eje X.
    - yaxis_title (str): Título para el eje Y.
    - total_accumulated (bool): Si es True, las barras se superponen. Si es False, las barras se apilan sumando sus valores.
    """
    # Crear la figura
    fig = go.Figure()

    barmode = 'overlay' if total_accumulated else 'stack'
    sum_values = {label: 0 for label in df[label_column]} if not total_accumulated else {}

    # Determinar el valor máximo para calcular el desplazamiento
    if total_accumulated:
        max_value = df[value_columns].max().max()
    else:
        max_value = df[value_columns].sum(axis=1).max()

    # Agregar cada serie de datos como una barra horizontal
    for column, color, legend_name in zip(value_columns, colors, legend_names):
        fig.add_trace(go.Bar(
            y=df[label_column],
            x=df[column],
            name=legend_name,
            marker_color=color,
            marker_line_color='black',
            marker_line_width=1.5,
            orientation='h'
        ))

        # Acumular valores para el modo apilado
        if not total_accumulated:
            for i, label in enumerate(df[label_column]):
                sum_values[label] += df[column].iloc[i]

    # Añadir etiquetas de valores a la derecha de cada barra
    for i, label in enumerate(df[label_column]):
        value_to_display = sum_values[label] if not total_accumulated else df[value_columns].max(axis=1).iloc[i]
        offset = value_to_display + (0.15 * max_value)  # Usar el 15% del valor máximo como desplazamiento
        fig.add_annotation(
            x=offset,
            y=label,
            text=str(value_to_display),
            showarrow=False,
            font=dict(
                family="Arial, sans-serif",
                size=16,
                color='black'
            )
        )

    # Actualizar diseño y estilo
    fig.update_layout(
        barmode=barmode,
        xaxis=dict(tickfont=dict(color='black'), title=xaxis_title),
        yaxis=dict(tickfont=dict(color='black'), title=yaxis_title),
        title=graph_title,
        legend=dict(x=1.05, y=1, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'),
        plot_bgcolor='rgba(255, 255, 255, 0)',
        paper_bgcolor='rgba(255, 255, 255, 0)',
        width=700,
        height=600
    )

    return fig










# ---------------------------------------------------------------------------------

# App Acordeón : módulo de FINANZAS

# PENDIENTE FILTRO DE LOCAL!!!!!!!!!!!!

def obtener_datos_finanzas(cliente_id):

    query_finanzas1 = """SELECT 
        COUNT(*) AS N_GASTOS, 
        SUM(g.importe) AS IMP_GASTOS_PERSONAL, 
        g.id_cliente, 
        g.id_local, 
        gf.id AS id_familia_gasto,
        gf.nombre AS nombre_familia_gasto 
    FROM 
        gastos g
    INNER JOIN 
        gastos_familias gf 
    ON 
        g.id_familia_gasto = gf.id
    WHERE 
        g.deleted_at IS NULL AND g.id_local IS NOT NULL
        AND g.id_cliente = %s
    GROUP BY 
        g.id_cliente, g.id_local, gf.id, gf.nombre
    ORDER BY 
        g.id_cliente, g.id_local
    """

    query_finanzas2 = """SELECT 
        g.id_cliente, 
        g.id_local, 
        fgpc.id AS id_familia,
        fgpc.nombre AS nombre_familia,
        SUM(g.importe) AS IMPORTE_TOTAL,
        COUNT(*) AS N_GASTOS
    FROM 
        gasto_personal_conceptos gpc 
    LEFT JOIN gastos g 
        ON gpc.id_gasto = g.id 
    INNER JOIN familia_gasto_personal_conceptos fgpc
        ON gpc.id_familia_concepto = fgpc.id
    WHERE 
        gpc.deleted_at IS NULL 
        AND g.deleted_at IS NULL 
        AND g.id_cliente IS NOT NULL AND g.id_local IS NOT NULL
        AND g.id_cliente = %s
    GROUP BY 
        g.id_cliente, g.id_local, id_familia, nombre_familia
    ORDER BY 
        g.id_cliente, g.id_local
    """

    query_finanzas3 = """SELECT COUNT(*) AS N_CIERRES, SUM(i.importe) AS IMPORTE_TOTAL, i.id_cliente, i.id_local, i.id_turno, t.nombre
            FROM ingresos i LEFT JOIN turnos t ON i.id_turno = t.id
            WHERE i.deleted_at IS NULL AND i.id_local IS NOT NULL AND i.id_cliente = %s
            GROUP BY i.id_cliente, i.id_local, i.id_turno
            ORDER BY i.id_cliente, i.id_local
    """
    query_recetas_inactivas = """
    SELECT COUNT(*) AS n_recetas_inactivas 
    FROM recetas r 
    WHERE r.id_cliente = %s AND r.activo = 0 AND r.deleted_at IS NULL
    """

    
    modulo_finanzas_df1 = bd_llamada(query_finanzas1, params_tuple=(cliente_id,))
    modulo_finanzas_df2 = bd_llamada(query_finanzas2, params_tuple=(cliente_id,))
    modulo_finanzas_df3 = bd_llamada(query_finanzas3, params_tuple=(cliente_id,))
    recetas_inactivas_df = bd_llamada(query_recetas_inactivas, params_tuple=(cliente_id,))

    return  modulo_finanzas_df1,  modulo_finanzas_df2,  modulo_finanzas_df3, recetas_inactivas_df









def obtener_datos_finanzas_1(cliente_id, fecha_rinicio, fecha_rfin):
    
    query_finanzas1 = """
    SELECT 
        COUNT(*) AS N_GASTOS, 
        SUM(g.importe) AS IMP_GASTOS_PERSONAL, 
        g.id_cliente, 
        g.id_local, 
        gf.id AS id_familia_gasto,
        gf.nombre AS nombre_familia_gasto 
    FROM 
        gastos g
    INNER JOIN 
        gastos_familias gf 
    ON 
        g.id_familia_gasto = gf.id
    WHERE 
        g.deleted_at IS NULL AND g.id_local IS NOT NULL
        AND g.id_cliente = %s
    GROUP BY 
        g.id_cliente, g.id_local, gf.id, gf.nombre
    ORDER BY 
        g.id_cliente, g.id_local
    """

    query_finanzas2 = """
    SELECT 
        g.id_cliente, 
        g.id_local, 
        fgpc.id AS id_familia,
        fgpc.nombre AS nombre_familia,
        SUM(g.importe) AS IMPORTE_TOTAL,
        COUNT(*) AS N_GASTOS
    FROM 
        gasto_personal_conceptos gpc 
    LEFT JOIN gastos g 
        ON gpc.id_gasto = g.id 
    INNER JOIN familia_gasto_personal_conceptos fgpc
        ON gpc.id_familia_concepto = fgpc.id
    WHERE 
        gpc.deleted_at IS NULL 
        AND g.deleted_at IS NULL 
        AND g.id_cliente IS NOT NULL
        AND g.id_local IS NOT NULL
        AND g.id_cliente = %s
    GROUP BY 
        g.id_cliente, g.id_local, id_familia, nombre_familia
    ORDER BY 
        g.id_cliente, g.id_local
    """

    # SELECT 
    #         COUNT(*) AS N_GASTOS, 
    #         SUM(g.importe) AS IMP_GASTOS_PERSONAL, 
    #         g.id_cliente, 
    #         g.id_local, 
    #         gf.id AS id_familia_gasto,
    #         gf.nombre AS nombre_familia_gasto 
    #     FROM 
    #         gastos g
    #     INNER JOIN 
    #         gastos_familias gf 
    #     ON 
    #         g.id_familia_gasto = gf.id
    #     WHERE 
    #         g.deleted_at IS NULL AND g.id_local IS NOT NULL
    #         AND g.id_cliente = 1
    #         AND g.id in (SELECT gpc.id_gasto from gasto_personal_conceptos gpc)
    #     GROUP BY 
    #         g.id_cliente, g.id_local, gf.id, gf.nombre
    #     ORDER BY 
    #         g.id_cliente, g.id_local;
            
        
    # SELECT 
    #         COUNT(*) AS N_GASTOS, 
    #         SUM(g.importe) AS IMP_GASTOS_PERSONAL, 
    #         g.id_cliente, 
    #         g.id_local, 
    #         gf.id AS id_familia_gasto,
    #         gf.nombre AS nombre_familia_gasto 
    #     FROM 
    #         gastos g
    #     INNER JOIN 
    #         gastos_familias gf 
    #     ON 
    #         g.id_familia_gasto = gf.id
    #     WHERE 
    #         g.deleted_at IS NULL AND g.id_local IS NOT NULL
    #         AND g.id_cliente = 1
    #     GROUP BY 
    #         g.id_cliente, g.id_local, gf.id, gf.nombre
    #     ORDER BY 
    #         g.id_cliente, g.id_local;

    query_finanzas12 = """SELECT 
        COUNT(*) AS N_GASTOS,
        SUM(g.importe) AS IMP_GASTOS_PERSONAL,
        g.id_cliente,
        g.id_local,
        gf.id AS id_familia_gasto,
        gf.nombre AS nombre_familia_gasto,
        CASE 
            WHEN g.id IN (SELECT gpc.id_gasto FROM gasto_personal_conceptos gpc) THEN 'SI'
            ELSE 'NO'
        END AS ASOCIADO_A_PERSONAL
    FROM 
        gastos g
    INNER JOIN 
        gastos_familias gf 
    ON 
        g.id_familia_gasto = gf.id
    WHERE 
        g.deleted_at IS NULL 
        AND g.id_local IS NOT NULL
        AND g.id_cliente = %s
        AND g.fecha >= %s AND g.fecha <= %s
    GROUP BY 
        g.id_cliente, g.id_local, gf.id, gf.nombre,
        CASE 
            WHEN g.id IN (SELECT gpc.id_gasto FROM gasto_personal_conceptos gpc) THEN 'SI'
            ELSE 'NO'
        END
    ORDER BY 
        g.id_cliente, g.id_local
    """




    # Consulta adicional para recetas inactivas
    query_recetas_inactivas = """
    SELECT COUNT(*) AS n_recetas_inactivas 
    FROM recetas r 
    WHERE r.id_cliente = %s AND r.activo = 0 AND r.deleted_at IS NULL
    """

    query_finanzas3 =  """
    SELECT 
        COUNT(*) AS N_CIERRES, 
        SUM(i.importe) AS IMPORTE_TOTAL, 
        i.id_cliente, 
        i.id_local, 
        i.id_turno, 
        t.nombre AS nombre_turno
    FROM 
        ingresos i 
    LEFT JOIN turnos t 
        ON i.id_turno = t.id
    WHERE 
        i.deleted_at IS NULL AND i.id_local IS NOT NULL AND i.id_cliente = %s
        AND i.fecha >= %s AND i.fecha <= %s
    GROUP BY 
        i.id_cliente, i.id_local, i.id_turno, t.nombre
    ORDER BY 
        i.id_cliente, i.id_local
    """

    #modulo_finanzas_df1 = bd_llamada(query_finanzas1, params_tuple=(cliente_id,))
    #modulo_finanzas_df2 = bd_llamada(query_finanzas2, params_tuple=(cliente_id,))
    modulo_finanzas_df12 = bd_llamada(query_finanzas12, params_tuple=(cliente_id, fecha_rinicio, fecha_rfin))
    modulo_finanzas_df3 = bd_llamada(query_finanzas3, params_tuple=(cliente_id, fecha_rinicio, fecha_rfin))
    # recetas_inactivas_df = bd_llamada(query_recetas_inactivas, params_tuple=(cliente_id,))

    print(modulo_finanzas_df12)
    print(modulo_finanzas_df3)
    return modulo_finanzas_df12,modulo_finanzas_df3






def crear_grafico_pastel_finanzas(df, columna_niveles, columna_valores, titulo, ancho_grafico, altura_grafico, color_map, nivel_negro):
    """
    Función para crear un gráfico de pastel con un sector extraído del centro.

    Parámetros:
    df (pd.DataFrame): DataFrame que contiene los datos.
    columna_niveles (str): Nombre de la columna que contiene los niveles o categorías.
    columna_valores (str): Nombre de la columna que contiene los valores a mostrar.
    titulo (str): Título del gráfico.
    ancho_grafico (int): Ancho del gráfico en píxeles.
    altura_grafico (int): Altura del gráfico en píxeles.
    color_map (dict): Diccionario que asigna un color HTML a cada cliente.
    nivel_negro: Corresponde al nivel que queremos mostrar siempre fuera como negro
    """

    # Asegurarnos de que las categorías siempre estén en el mismo orden basado en color_map
    df[columna_niveles] = pd.Categorical(df[columna_niveles], categories=color_map.keys(), ordered=True)
    df_sorted = df.sort_values(by=columna_niveles)

    # Etiquetas y valores para el gráfico de pastel
    labels = df_sorted[columna_niveles].tolist()
    values = df_sorted[columna_valores].tolist()

    # Asignación de colores basados en el color_map proporcionado
    colores_usados = [color_map.get(label, '#000000') for label in labels]

    # Configuración para 'explosionar' solo el sector del 'nivel_negro' si está presente
    pull = [0.2 if label == nivel_negro else 0 for label in labels]
    pull = 0.1

    # Creación del gráfico de pastel con borde negro
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        pull=pull, 
        marker=dict(colors=colores_usados, line=dict(color='black', width=1.5))
    )])

    fig.update_layout(
        title=titulo,
        width=ancho_grafico,
        height=altura_grafico,
        legend=dict(
            traceorder="normal"
        )
    )

    return fig



def plot_barras_ingresos(valores, niveles, colores, titulo, ancho_grafico, altura_grafico):
    """
    Función para crear un gráfico de barras horizontal en Plotly.

    Parámetros:
    valores (list): Lista de valores a mostrar en el gráfico.
    niveles (list): Lista de niveles o categorías correspondientes a los valores.
    colores (list): Lista de colores HTML para cada nivel.
    titulo (str): Título del gráfico.
    ancho_grafico (int): Ancho del gráfico en píxeles.
    altura_grafico (int): Altura del gráfico en píxeles.
    """

    # Creación del gráfico de barras horizontal
    fig = go.Figure(data=[go.Bar(
        x=valores, 
        y=niveles, 
        orientation='h',  # 'h' para gráfico horizontal
        marker=dict(
            color=colores,  # Aplicar los colores a las barras
            line=dict(color='black', width=1.5)  # Contorno negro con grosor de 1.5 píxeles
        )
    )])

    # Configuración adicional del gráfico
    fig.update_layout(
        title=titulo,
        xaxis_title='Valores',
        yaxis_title='Niveles',
        width=ancho_grafico,  # Ancho del gráfico
        height=altura_grafico  # Altura del gráfico
    )

    # Devolver el gráfico
    return fig

import plotly.express as px

def g_sectores_agrupado_gastos(df, columna_niveles, columna_valores, titulo, ancho_grafico, altura_grafico):
    """
    Función para crear un gráfico de pastel con un sector extraído del centro y agrupación de categorías pequeñas.

    Parámetros:
    df (pd.DataFrame): DataFrame que contiene los datos.
    columna_niveles (str): Nombre de la columna que contiene los niveles o categorías.
    columna_valores (str): Nombre de la columna que contiene los valores a mostrar.
    titulo (str): Título del gráfico.
    ancho_grafico (int): Ancho del gráfico en píxeles.
    altura_grafico (int): Altura del gráfico en píxeles.
    """

    # Agrupar los valores que tienen el mismo nivel
    df = df.groupby(columna_niveles, as_index=False).sum()

    # Agrupamos categorías menores al 5% si hay más de 4 categorías
    if len(df) > 4:
        total_valor = df[columna_valores].sum()
        df['porcentaje'] = df[columna_valores] / total_valor * 100
        df_small = df[df['porcentaje'] < 2]
        if not df_small.empty:
            df_large = df[df['porcentaje'] >= 2]
            otros_valor = df_small[columna_valores].sum()
            
            df_otros = pd.DataFrame({columna_niveles: ['Otros'], columna_valores: [otros_valor]})
            df = pd.concat([df_large, df_otros], ignore_index=True)

    # Etiquetas y valores para el gráfico de pastel
    labels = df[columna_niveles].tolist()
    values = df[columna_valores].tolist()

    # Asignación de colores utilizando una paleta de colores predeterminada de Plotly
    colores_usados = px.colors.qualitative.Plotly * ((len(labels) // len(px.colors.qualitative.Plotly)) + 1)

    # Creación del gráfico de pastel con borde negro
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        marker=dict(colors=colores_usados[:len(labels)], line=dict(color='black', width=1.5))
    )])

    fig.update_layout(
        title=titulo,
        width=ancho_grafico,
        height=altura_grafico,
        legend=dict(
            traceorder="normal"
        )
    )

    return fig


# -----------------------------------------------------------------------------------------------------------------------------------------------------

# App Acordeón : módulo de OCR

# def bd_llamada_ocr(query, params_tuple):
#     server_ocr = 'yurestazure.database.windows.net'
#     database_ocr = 'yuBDazure'
#     username_ocr = 'CloudSA19f38018'
#     password_ocr = '65gh_34ddf'  # Debes proporcionar el password

#     connection_string_ocr = f'mssql+pyodbc://{username_ocr}:{password_ocr}@{server_ocr}/{database_ocr}?driver=ODBC+Driver+17+for+SQL+Server'
#     engine_ocr = create_engine(connection_string_ocr)

#     # Ejecutar la consulta y obtener un DataFrame de pandas
#     df = pd.read_sql_query(text(query), engine_ocr, params=params_tuple)
#     return df


import pandas as pd
from sqlalchemy import create_engine, text
import pymssql


# def print_keys():
#     print("\n\n\nHost OCR: ", os.getenv("OCR_HOST"), "\n\n\n", 'HOST SQL SERVER: ', os.getenv("MYSQL_HOST"))
#     print("\n\n\nUser OCR: ", os.getenv("OCR_USER"), "\n\n\n", 'USER SQL SERVER: ', os.getenv("MYSQL_USER"))
#     print("\n\n\nPassword OCR: ", os.getenv("OCR_PASSWORD"), "\n\n\n", 'PASSWORD SQL SERVER: ', os.getenv("MYSQL_PASSWORD"))
#     print("\n\n\nDB OCR: ", os.getenv("OCR_DB"), "\n\n\n", 'DB SQL SERVER: ', os.getenv("MYSQL_DB"))
#     print("\n\n\nPORT SQL SERVER: ", os.getenv("MYSQL_PORT"))
    

def bd_llamada_ocr(query, params_tuple):
    OCR_HOST = os.getenv("OCR_HOST")
    OCR_USER = os.getenv("OCR_USER")
    OCR_PASSWORD = os.getenv("OCR_PASSWORD")
    OCR_DB = os.getenv("OCR_DB")
    # Cambiar la cadena de conexión para usar pymssql
    # connection_string_ocr = f"mssql+pymssql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
    connection_string_ocr = f"mssql+pyodbc://{OCR_USER}:{OCR_PASSWORD}@{OCR_HOST}/{OCR_DB}?driver=ODBC+Driver+17+for+SQL+Server"

    engine_ocr = create_engine(connection_string_ocr)

    # Ejecutar la consulta SQL y obtener un DataFrame
    df = pd.read_sql_query(text(query), engine_ocr, params=params_tuple)
    return df


def ids_clientes_ocr():
    query_clientes_ocr = """select id_cliente from clientes"""
    df_procesados = bd_llamada_ocr(query_clientes_ocr, params_tuple=())
    lista_ids_cliente = set(df_procesados['id_cliente'].tolist())
    return lista_ids_cliente



def obtener_datos_ocr(cliente_id, fecha_rinicio, fecha_rfin):
    
       # Número de albaranes registrados # Número de modelos utilizados # Albaranes que han pasado correctamente
       # Albaranes donde ha fallado la TABLA DE ÍTEMS # Albaranes donde ha fallado el ENCABEZADO

    cliente_id = int(cliente_id)
    
    query_ocr_1 = """SELECT (SELECT COUNT(DISTINCT a.id) FROM albaranes a WHERE a.id_cliente = :id_cliente) AS N_ALBARANES_OCR,
       (SELECT COUNT(DISTINCT a.id_model) FROM albaranes a WHERE a.id_cliente = :id_cliente) AS N_MODELOS_OCR,
       (SELECT COUNT(*) FROM albaranes a WHERE a.incidencias = '' AND a.encabezado_correcto = 1 AND a.id_cliente = :id_cliente) AS N_ALBARANES_DIRECTOS,
       (SELECT COUNT(*) FROM (SELECT DISTINCT ir.id_albaran FROM item_revisado ir 
              WHERE (ir.producto = 1 OR ir.cantidad = 1 OR ir.precio_unitario = 1 OR ir.precio = 1 OR ir.descuento = 1)
                     AND ir.id_albaran IN (SELECT DISTINCT id FROM albaranes a WHERE a.id_cliente = :id_cliente AND a.encabezado_correcto = 1) GROUP BY ir.id_albaran) as q) AS F_TABLA_ITEMS,
       (SELECT COUNT(DISTINCT ar.id_albaran) FROM alb_revisado ar 
              WHERE (ar.fecha = 1 OR ar.subtotal = 1 OR ar.iva = 1 OR ar.total = 1 OR ar.descuentos = 1 OR ar.otros_impuestos = 1)
              AND ar.id_albaran IN (SELECT DISTINCT id FROM albaranes a WHERE a.id_cliente = :id_cliente AND a.encabezado_correcto = 1)) AS F_ENCABEZADO"""
    print(cliente_id)

    query_ocr_1 = """SELECT 
    (SELECT COUNT(DISTINCT a.id) 
     FROM albaranes a 
     INNER JOIN logs l ON a.id_albaran = l.id_albaran
     WHERE a.id_cliente = :id_cliente 
       AND l.fecha_creacion >= :fecha_inicio 
       AND l.fecha_creacion <= :fecha_fin) AS N_ALBARANES_OCR,

    (SELECT COUNT(DISTINCT a.id_model) 
     FROM albaranes a 
     INNER JOIN logs l ON a.id_albaran = l.id_albaran
     WHERE a.id_cliente = :id_cliente 
       AND l.fecha_creacion >= :fecha_inicio 
       AND l.fecha_creacion <= :fecha_fin) AS N_MODELOS_OCR,

    (SELECT COUNT(*) 
     FROM albaranes a 
     INNER JOIN logs l ON a.id_albaran = l.id_albaran
     WHERE a.incidencias = '' 
       AND a.encabezado_correcto = 1 
       AND a.id_cliente = :id_cliente 
       AND l.fecha_creacion >= :fecha_inicio 
       AND l.fecha_creacion <= :fecha_fin) AS N_ALBARANES_DIRECTOS,

    (SELECT COUNT(*) 
     FROM (
        SELECT DISTINCT ir.id_albaran 
        FROM item_revisado ir 
        INNER JOIN albaranes a ON ir.id_albaran = a.id
        INNER JOIN logs l ON a.id_albaran = l.id_albaran
        WHERE (ir.producto = 1 OR ir.cantidad = 1 OR ir.precio_unitario = 1 OR ir.precio = 1 OR ir.descuento = 1)
          AND a.id_cliente = :id_cliente 
          AND a.encabezado_correcto = 1 
          AND l.fecha_creacion >= :fecha_inicio 
          AND l.fecha_creacion <= :fecha_fin
        GROUP BY ir.id_albaran
     ) as q) AS F_TABLA_ITEMS,

    (SELECT COUNT(DISTINCT ar.id_albaran) 
     FROM alb_revisado ar 
     INNER JOIN albaranes a ON ar.id_albaran = a.id
     INNER JOIN logs l ON a.id_albaran = l.id_albaran
     WHERE (ar.fecha = 1 OR ar.subtotal = 1 OR ar.iva = 1 OR ar.total = 1 OR ar.descuentos = 1 OR ar.otros_impuestos = 1)
       AND a.id_cliente = :id_cliente 
       AND a.encabezado_correcto = 1 
       AND l.fecha_creacion >= :fecha_inicio 
       AND l.fecha_creacion <= :fecha_fin) AS F_ENCABEZADO
       """
    
    ocr_df1 = bd_llamada_ocr(query_ocr_1, params_tuple={'id_cliente': cliente_id,  'fecha_inicio':fecha_rinicio, 'fecha_fin': fecha_rfin})

        # Modelos que más fallan en el último mes

    query_ocr_2 = """SELECT a.id_model, COUNT(a.id_model) AS N_FALLOS_MODELO FROM albaranes a
              INNER JOIN logs l ON a.id_albaran = l.id_albaran
              WHERE l.fecha_creacion >= :fecha_inicio AND l.fecha_creacion <= :fecha_fin
              AND a.incidencias LIKE '%El modelo no reco%' AND a.id_cliente = :id_cliente 
              GROUP BY a.id_model ORDER BY N_FALLOS_MODELO DESC"""
    
    ocr_df2 = bd_llamada_ocr(query_ocr_2, params_tuple={'id_cliente': cliente_id, 'fecha_inicio':fecha_rinicio,'fecha_fin': fecha_rfin})
       
       # Incidencias encontradas

    query_ocr_3 = """SELECT COUNT(a.incidencias) AS N, a.incidencias FROM albaranes a
       INNER JOIN logs l ON a.id_albaran = l.id_albaran
       WHERE l.fecha_creacion >= :fecha_inicio AND l.fecha_creacion <= :fecha_fin
       AND PATINDEX('%[a-zA-Z]%', a.incidencias) > 0 
       AND a.id_cliente = :id_cliente GROUP BY a.incidencias ORDER BY N DESC""" 
    ocr_df3 = bd_llamada_ocr(query_ocr_3, params_tuple={'id_cliente': cliente_id, 'fecha_inicio':fecha_rinicio,'fecha_fin': fecha_rfin})

    return ocr_df1, ocr_df2, ocr_df3


def procesar_incidencias(df):
    # Verificar si el DataFrame está vacío
    if df.empty:
        # Retornar un DataFrame vacío con las columnas esperadas
        return pd.DataFrame(columns=['incidencia', 'TOTAL'])

    incidencias_expandido = []

    # Iterar sobre cada fila del DataFrame
    for i, row in df.iterrows():
        # Quitar los corchetes y las comillas
        incidencias = row['incidencias'].strip("[]").replace("'", "")
        
        # Separar por comas y eliminar espacios en blanco
        incidencias_lista = [x.strip() for x in incidencias.split(",")]

        # Crear un registro para cada incidencia
        for incidencia in incidencias_lista:
            incidencias_expandido.append({'incidencia': incidencia, 'TOTAL': row['N']})

    # Crear un nuevo DataFrame con los datos descompuestos
    nuevo_df = pd.DataFrame(incidencias_expandido)

    # Agrupar por 'incidencia' y sumar los valores de 'TOTAL'
    resultado_df = nuevo_df.groupby('incidencia', as_index=False).sum()
    resultado_df = resultado_df.sort_values(by='TOTAL', ascending=False).reset_index(drop=True)

    return resultado_df


def filtro_ultima_version(df, columna_fallos):
    # Crear una columna para extraer el número de versión y otra para el proveedor
    df['version'] = df[columna_fallos].str.extract(r'v(\d+)-', expand=False)
    df['proveedor'] = df[columna_fallos].str.extract(r'-(\w+)', expand=False)
    
    # Convertir la columna de versión a numérica, manejando los valores NaN para los casos que no comienzan con "v"
    df['version'] = pd.to_numeric(df['version'], errors='coerce')
    
    # Identificar las filas que tienen versión numérica (empiezan con "v")
    df_con_version = df.dropna(subset=['version'])
    
    # Ordenar por proveedor y versión
    df_con_version = df_con_version.sort_values(by=['proveedor', 'version'], ascending=[True, False])
    
    # Eliminar duplicados, manteniendo solo la última versión para cada proveedor
    df_con_version = df_con_version.drop_duplicates(subset=['proveedor'], keep='first')
    
    # Unir las filas con versión y las que no tienen versión
    df_final = pd.concat([df_con_version, df[df['version'].isna()]]).reset_index(drop=True)
    df_final = df_final.drop(columns=['proveedor', 'version'])
    df_final = df_final.sort_values(by=['N_FALLOS_MODELO'], ascending= False)

    return df_final


def create_donut_chart_ocr(chart_title, center_text, colores, labels, values):
    #labels = ['ACTIVO', 'NO ACTIVO']
    #values = [active_count, inactive_count]
    #colors = ['green', 'red']  # Verde para 'ACTIVO' y rojo para 'NO ACTIVO'

    # Calcula el total para el número central
    total = sum(values)

    # Crear el gráfico de dona
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=.5, 
        marker=dict(
            colors=colores,
            line=dict(color='black', width=1)  # Bordes negros más finos
        )
    )])

    # Personaliza y agregar el número en el centro
    fig.update_layout(
        title_text=chart_title,
        annotations=[dict(text=f'{total}<br>{center_text}', x=0.5, y=0.5, font_size=15, showarrow=False, font=dict(color='black'))],
        width=450,  # Ajustar el ancho de la figura
        height=350,  # Ajustar la altura de la figura
        margin=dict(l=50, r=50, t=50, b=50),  # Ajustar los márgenes
        showlegend=True  # Mostrar la leyenda
    )

    # Devolver la figura en lugar de mostrarla directamente
    return fig



# SELECT 
#     COUNT(CASE WHEN t.id_familia = 4 AND t.fecha_inicio >= '2024-08-01' AND t.fecha_inicio <= '2024-08-31' THEN 1 END) AS AVERIAS,
#     COUNT(CASE WHEN t.id_familia = 4 AND t.fecha_inicio >= '2024-08-01' AND t.fecha_inicio <= '2024-08-31' THEN 1 END) AS AVERIAS_TERMINADAS,
#     COUNT(CASE WHEN t.id_familia = 3 AND t.fecha_inicio >= '2024-08-01' AND t.fecha_inicio <= '2024-08-31' THEN 1 END) AS AVISOS,
#     COUNT(CASE WHEN t.id_familia = 3 AND t.fecha_inicio >= '2024-08-01' AND t.fecha_inicio <= '2024-08-31' THEN 1 END) AS AVERIAS,
#     COUNT(CASE WHEN t.id_familia = 1 AND t.fecha_inicio >= '2024-08-01' AND t.fecha_inicio <= '2024-08-31' THEN 1 END) AS EVENTOS,
#     COUNT(CASE WHEN t.id_familia = 1 AND t.fecha_inicio >= '2024-08-01' AND t.fecha_inicio <= '2024-08-31' THEN 1 END) AS AVERIAS,
#     COUNT(CASE WHEN t.id_familia = 2 AND t.fecha_inicio >= '2024-08-01' AND t.fecha_inicio <= '2024-08-31' THEN 1 END) AS REUNIONES,
#     COUNT(CASE WHEN t.id_familia = 2 AND t.fecha_inicio >= '2024-08-01' AND t.fecha_inicio <= '2024-08-31' THEN 1 END) AS AVERIAS,
#     COUNT(CASE WHEN t.id_familia = 10 THEN 1 END) AS CHATS,
#     COUNT(CASE WHEN t.id_familia = 10 THEN 1 END) AS AVERIAS
# FROM tareas t 
# WHERE t.id_cliente = 1 
#   AND t.deleted_at IS NULL;


# -- Máquinas
# SELECT * FROM maquinas m WHERE m.id_cliente = 1 AND m.deleted_at IS NULL;

# -- Productos
# SELECT * FROM apccc_productos ap ;
 
# -- Proveedores
# SELECT * FROM apccc_proveedores ap WHERE ap.id_cliente = 1 AND ap.deleted_at IS NULL;
# -- Recurrencias
# SELECT * FROM asignaciones_apcc WHERE id_cliente = 1 AND deleted_at IS NULL ORDER BY nombre;


#  -- Averías
# SELECT * FROM tareas t WHERE t.id_familia = 4 AND t.id_cliente = 1 AND t.deleted_at IS NULL AND YEAR(t.fecha_inicio) = 2024 AND MONTH(t.fecha_inicio) = 8;

# -- Avisos
# SELECT * FROM tareas t WHERE t.id_familia = 3 AND t.id_cliente = 1 AND t.deleted_at IS NULL AND YEAR(t.fecha_inicio) = 2024;

# -- Eventos
# SELECT * FROM tareas t WHERE t.id_familia = 1 AND t.id_cliente = 1 AND t.deleted_at IS NULL AND YEAR(t.fecha_inicio) = 2024;

# -- Reuniones
# SELECT * FROM tareas t WHERE t.id_familia = 2 AND t.id_cliente = 1 AND t.deleted_at IS NULL AND YEAR(t.fecha_inicio) = 2024;

# -- Chats
# SELECT * FROM tareas t WHERE t.id_familia = 10 AND t.id_cliente = 1 AND t.deleted_at IS NULL AND YEAR(t.fecha_inicio) = 2024;


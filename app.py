import subprocess
import sys

def install_requirements():
    try:
        # Comando para instalar las librerías de requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError as e:
        print(f"Error al instalar dependencias: {e}")
        sys.exit(1)

# Llamada a la función para instalar las dependencias
install_requirements()



import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
from funciones import *
from shiny import App, Inputs, Outputs, Session, render, ui
import json
from jinja2 import Template
import matplotlib.pyplot as plt
import time
import datetime

print_host()

# Obtener los datos de clientes y sus módulos
df_clientes, lista_clientes = lista_clientes()

# Fechas límite para el selector de rango de fechas
fecha_inicio_rango = datetime.date(2024, 8, 1)
fecha_actual = datetime.date.today()
fecha_actual_rango = datetime.date.today()



# Función para generar un gráfico de barras con Plotly
def generate_example_chart():
    fig = px.bar(x=['A', 'B', 'C'], y=[10, 20, 30], title='Ejemplo de Gráfico de Barras')
    return fig

# Función para generar una tabla de ejemplo con Plotly
def generate_example_table():
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Producto', 'Ventas'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[['Producto A', 'Producto B', 'Producto C'],
                           [100, 150, 200]],
                   fill_color='lavender',
                   align='left'))
    ])
    return fig

# Función para generar un scatter plot de ejemplo
def generate_scatter_plot():
    x = np.random.rand(100)
    y = np.random.rand(100)
    fig = px.scatter(x=x, y=y, title='Scatter Plot de Puntos Verdes')
    return fig


# ---------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------------



# Función para generar los paneles de acordeón
def make_items(cliente_data, fecha_inicio_rango, fecha_fin, lista_modulos):
    if cliente_data.empty:
        return [ui.accordion_panel("Error", "No hay datos para el cliente seleccionado.")]

    items = []
    # AÑADIR ALMACENES 
    #modulos_presentes = ['cocina',  'rrhh', 'almacenes', 'tar', 'dashboards', 'compras', 'finanzas', 'ocr','formacion',  'kds', 'basicos', 'amparo', 'documentos']
    modulos_presentes = ['cocina'] 
    modulos_presentes = lista_modulos
    for modulo in modulos_presentes:
        if modulo in cliente_data.columns:
            if cliente_data[modulo].iloc[0] == 1:
                estado = "contratado"
                background_color = "#D1FAE5"  # Verde esmeralda claro
                content = "Módulo contratado" 
                if modulo == 'kds' or modulo == 'amparo':
                    # URL directo a la imagen
                    image_url = "https://yurest-prod.s3.eu-west-3.amazonaws.com/perfil_cliente/PjlNWpwa2IHQv0PFnb2VgVTTqTdhpIao5PmgxGFj.svg"
                    historia = f"Esta es la historia de {modulo.upper()}: YUREEEEEEEST"
                    items.append(
                        ui.accordion_panel(
                            f"{modulo}".upper(),
                            ui.div(
                                ui.p(content),
                                ui.div(
                                    ui.HTML(f"<img src='{image_url}' style='width:100%; max-width:400px;'>"),
                                    ui.p(historia),
                                    style=f"background-color: {background_color}; padding: 10px; border-radius: 5px;"  # Estilo de fondo dependiendo del estado
                                )
                            ),
                            class_="accordion-panel-enabled"
                        )   
                    )    

# --------------------------------------------------------------------------------------------------------------------------------------
                    
                elif modulo == 'rrhh':
                    start_rrhh = time.time()
                    modulo_rrhh_df1, modulo_rrhh_df2, modulo_rrhh_df3,  modulo_rrhh_df4, modulo_rrhh_df5, modulo_rrhh_df6, modulo_rrhh_df7 = obtener_datos_rrhh(cliente_data['id'].iloc[0], fecha_inicio_rango, fecha_fin)
  
                    # Obtener los datos de rrhh
                    n_empleados = modulo_rrhh_df1['n_empleados'].iloc[0]
                    n_fichajes = modulo_rrhh_df4['n_fichajes'].iloc[0]
                    n_sol_vac = modulo_rrhh_df5['n_sol_vac'].iloc[0]
                    n_sol_vac_acept = modulo_rrhh_df5['n_sol_vac_acept'].iloc[0]
                    n_planificaciones = modulo_rrhh_df6['n_planificaciones_hoy'].iloc[0]
                    solicitudes_no_acep = n_sol_vac - n_sol_vac_acept


                    df_3 = modulo_rrhh_df3
                    fig = plot_vacation_distribution(df_3)
                    fig_html = fig_to_html(fig)

                    # Gráfico circular: solicitudes de vacaciones aceptadas y no aceptadas
                    #fig_circular = px.pie(values=[n_sol_vac_acept, solicitudes_no_acep], names=['Aceptadas', 'No aceptadas'], title='Solicitudes de vacaciones')
                    #fig_roles_html = plotly_to_html(fig_circular)

                    # Tabla: roles configurados
                    roles = modulo_rrhh_df2
                    tabla_html = roles.to_html(index=False, classes='table table-striped')
                    fig_roles = px.bar(roles, x='N_EMPLEADOS', y='ROL', orientation='h', title='Número de Empleados por Rol')
                    fig_roles.update_layout(
                        height=600,  # Ajusta la altura del gráfico
                        margin=dict(l=150, r=20, t=50, b=50),  # Ajusta los márgenes
                        yaxis=dict(tickangle=0)  # Rotar las etiquetas del eje y a 0 grados (horizontal)
                    )
                    
                    plot_tipo_ausencia = g_sectores_agrupado_gastos(modulo_rrhh_df7, columna_niveles = 'tipo_ausencia', columna_valores = 'N', titulo = 'Distribución de Ausencias', ancho_grafico = 700 , altura_grafico = 600)
                    plot_tipo_ausencia = plotly_to_html(plot_tipo_ausencia)

                    donut_vacaciones = create_donut_chart('Solicitud de Vacaciones', '', colores = ['brown', 'red'], labels = ['Aceptadas', 'No Aceptadas'], values = [n_sol_vac_acept, solicitudes_no_acep])
                    donut_vacaciones = plotly_to_html(donut_vacaciones)

                    gbarras_roles_empleados = plot_barras_compras(roles,label_column='ROL',
                                                value_columns = ['N_EMPLEADOS'],
                                                colors = ['brown'],
                                                legend_names = ['A'],
                                                graph_title = 'Empleados por Rol',
                                                xaxis_title = 'Número de Empleados',
                                                yaxis_title = 'Rol',
                                                total_accumulated=False
                                            )
                    gbarras_roles_empleados = plotly_to_html(gbarras_roles_empleados)

                    # # CSS para ajustar el ancho de las columnas de la tabla
                    # tabla_css = """
                    # <style>
                    # .table {
                    #     width: auto;
                    #     table-layout: auto;
                    #     margin: auto;
                    # }
                    # .table th, .table td {
                    #     white-space: nowrap;
                    #     overflow: hidden;
                    #     text-overflow: ellipsis;
                    #     max-width: 150px;
                    # }
                    # </style>
                    # """

                    # # Tarjetas
                    # tarjeta_empleados = f"""
                    # <div style="background-color: #D1FAE5; padding: 20px; border-radius: 10px; width: 200px; text-align: center;">
                    #     <h4>Número de empleados: {n_empleados}</h4>
                    # </div>
                    # """

                    # tarjeta_fichajes = f"""
                    # <div style="background-color: #D1FAE5; padding: 20px; border-radius: 10px; width: 200px; text-align: center;">
                    #     <h4>Número de fichajes: {n_fichajes}</h4>
                    # </div>
                    # """

                    # tarjeta_planificaciones = f"""
                    # <div style="background-color: #D1FAE5; padding: 20px; border-radius: 10px; width: 200px; text-align: center;">
                    #     <h4>Número de planificaciones HOY: {n_planificaciones}</h4>
                    # </div>
                    # """

                    # tarjeta_vacaciones = f"""
                    # <div style="background-color: #D1FAE5; padding: 20px; border-radius: 10px; width: 200px; text-align: center;">
                    #     <h4>Número de solicitud de vacaciones: {n_sol_vac}</h4>
                    # </div>
                    # """
                    # tarjeta_vacaciones_distribucion = f"""
                    # <div style="background-color: #D1FAE5; padding: 20px; border-radius: 10px; width: 100%;">
                    #    fig_roles_html {fig_html}
                    # </div>
                    # """

                    # items.append(
                    #     ui.accordion_panel(
                    #         f"{modulo}".upper(),
                    #         ui.div(
                    #             ui.p(content),
                    #             ui.div(
                    #                 ui.HTML(tarjeta_empleados),
                    #                 ui.HTML(tarjeta_fichajes),
                    #                 ui.HTML(tarjeta_planificaciones),
                    #                 ui.HTML(tarjeta_vacaciones),
                    #                 style="display: flex; flex-wrap: wrap; gap: 20px; align-items: center;"  # Alinea los elementos horizontalmente y agrega espacio entre ellos
                    #             ),
                    #             ui.div(
                    #                 ui.HTML(fig_circular.to_html(full_html=False)),
                    #                 ui.HTML(fig_roles.to_html(full_html=False)),
                    #                 ui.HTML(tabla_css + tabla_html),  # Incluye el CSS y la tabla HTML
                    #                 style="display: flex; flex-wrap: wrap; gap: 20px; align-items: center;"  # Alinea los elementos horizontalmente y agrega espacio entre ellos
                    #             ),
                    #             ui.div(
                    #                 ui.HTML(tarjeta_vacaciones_distribucion),
                    #                 style="margin-top: 20px;"
                    #             )
                    #         ),
                    #         class_="accordion-panel-enabled"
                    #     )
                    # )

                    fondo_marron_claro = "#F5C4AD"

                    # Tarjeta superior con valores y gráficos gbarras_roles_empleados y donut_vacaciones
                    tarjeta_superior = f"""
                    <div style="background-color: {fondo_marron_claro}; padding: 20px; border-radius: 10px; display: flex; flex-direction: column; gap: 20px;">
                        <!-- Sección de tarjetas con valores -->
                        <div style="display: flex; gap: 20px;">
                            <!-- Tarjeta 1: Número de empleados -->
                            <div style="flex: 1;">
                                <div style="padding: 10px; background-color: #FF5733; border-radius: 5px; height: 100px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                    <h5 style="font-weight: bold; text-align: center; color: white;">NÚMERO DE EMPLEADOS</h5>
                                    <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{n_empleados}</p>
                                </div>
                            </div>
                            
                            <!-- Tarjeta 2: Número de fichajes -->
                            <div style="flex: 1;">
                                <div style="padding: 10px; background-color: #FFB200; border-radius: 5px; height: 100px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                    <h5 style="font-weight: bold; text-align: center; color: white;">NÚMERO DE FICHAJES</h5>
                                    <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{n_fichajes}</p>
                                </div>
                            </div>

                            <!-- Tarjeta 3: Número de solicitudes de vacaciones -->
                            <div style="flex: 1;">
                                <div style="padding: 10px; background-color: #000000; border-radius: 5px; height: 100px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                    <h5 style="font-weight: bold; text-align: center; color: white;">SOLICITUDES DE VACACIONES</h5>
                                    <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{n_sol_vac}</p>
                                </div>
                            </div>

                            <!-- Tarjeta 4: Número de planificaciones -->
                            <div style="flex: 1;">
                                <div style="padding: 10px; background-color: #821DB8; border-radius: 5px; height: 100px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                    <h5 style="font-weight: bold; text-align: center; color: white;">NÚMERO DE PLANIFICACIONES HOY</h5>
                                    <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{n_planificaciones}</p>
                                </div>
                            </div>
                        </div>

                        <!-- Sección de gráficos: gbarras_roles_empleados y donut_vacaciones -->
                        <div style="display: flex; gap: 20px; justify-content: space-around; align-items: center;">
                            <div style="flex: 1;">
                                {gbarras_roles_empleados}
                            </div>
                            <div style="flex: 1;">
                                {donut_vacaciones}
                            </div>
                        </div>
                    </div>
                    """

                    # Tarjeta inferior con los gráficos restantes
                    tarjeta_inferior = f"""
                    <div style="background-color: {fondo_marron_claro}; padding: 20px; border-radius: 10px; display: flex; flex-direction: row; gap: 20px;">
                        <!-- Gráfico de tipos de ausencia -->
                        <div style="flex: 1; display: flex; justify-content: center;">
                            {plot_tipo_ausencia}
                        </div>

                        <!-- Gráfico adicional -->
                        <div style="flex: 1; display: flex; justify-content: center;">
                            {fig_html}
                        </div>
                    </div>
                    """


                    # Añadir las tarjetas al panel de acordeón, colocando las tarjetas una encima de la otra
                    items.append(
                        ui.accordion_panel(
                            f"{modulo}".upper(),
                            ui.div(
                                ui.p(content),
                                ui.div(
                                    ui.div(ui.HTML(tarjeta_superior), style="width: 100%; margin-bottom: 20px;"),  # Tarjeta superior
                                    ui.div(ui.HTML(tarjeta_inferior), style="width: 100%;"),  # Tarjeta inferior
                                    style="display: flex; flex-direction: column; gap: 20px;"  # Colocar las tarjetas en una columna con espacio entre ellas
                                ),
                                style="width: 100%;"  # Ocupa todo el ancho
                            ),
                            class_="accordion-panel-enabled"
                        )
                    )


                    end_rrhh = time.time()
                    time_rrhh = end_rrhh - start_rrhh
                    print("T-RRHH:", time_rrhh)
                    


# --------------------------------------------------------------------------------------------------------------------------------
                
                elif modulo == 'cocina':
                    start_cocina = time.time()
                    modulo_cocina_df1 = obtener_datos_cocina(cliente_data['id'].iloc[0])

                    # Obtener los datos de rrhh
                    n_recetas_activas = modulo_cocina_df1['n_recetas_activas'].iloc[0]
                    n_recetas_inactivas = modulo_cocina_df1['n_recetas_inactivas'].iloc[0]
                    n_recetas = n_recetas_activas + n_recetas_inactivas

                    n_fichas_tecnicas_activas = modulo_cocina_df1['n_fichas_tecnicas_activas'].iloc[0]
                    n_fichas_tecnicas_inactivas = modulo_cocina_df1['n_fichas_tecnicas_inactivas'].iloc[0]
                    n_fichas_tecnicas = n_fichas_tecnicas_activas + n_fichas_tecnicas_inactivas

                    # n_fichas_tecnicas = modulo_cocina_df1['n_fichas_tecnicas'].iloc[0]
                    n_elaboraciones_activas = modulo_cocina_df1['n_elaboraciones_activas'].iloc[0]
                    n_elaboraciones_inactivas = modulo_cocina_df1['n_elaboraciones_inactivas'].iloc[0]
                    n_elaboraciones = n_elaboraciones_inactivas + n_elaboraciones_activas

                    n_etiquetas_favoritas = modulo_cocina_df1['n_etiquetas_favoritas'].iloc[0]
                    n_etiquetas_nofavoritas = modulo_cocina_df1['n_etiquetas_nofavoritas'].iloc[0]
                    n_etiquetas = n_etiquetas_favoritas + n_etiquetas_nofavoritas

                    # fig_circular_recetas = px.pie(values=[n_recetas_activas, n_recetas_inactivas], names=['Recetas activas', 'Recetas inactivas'], title='Estado de las recetas')
                    
                    tarjeta_recetas = f"""
                    <div style="background-color: #D1FAE5; padding: 20px; border-radius: 10px; width: 200px; text-align: center;">
                        <h4>Número de recetas: {n_recetas}</h4>
                    </div>
                    """

                    tarjeta_fichas_tecnicas = f"""
                    <div style="background-color: #D1FAE5; padding: 20px; border-radius: 10px; width: 200px; text-align: center;">
                        <h4>Número de fichas técnicas: {n_fichas_tecnicas}</h4>
                    </div>
                    """

                    tarjeta_elaboraciones = f"""
                    <div style="background-color: #D1FAE5; padding: 20px; border-radius: 10px; width: 200px; text-align: center;">
                        <h4>Número de elaboraciones: {n_elaboraciones}</h4>
                    </div>
                    """

                    tarjeta_etiquetas = f"""
                    <div style="background-color: #D1FAE5; padding: 20px; border-radius: 10px; width: 200px; text-align: center;">
                        <h4>Número de etiquetas: {n_etiquetas}</h4>
                    </div>
                    """

                    donut_recetas = create_donut_chart('', 'RECETAS', colores = ['green', 'red'], labels = ['ACTIVO', 'NO ACTIVO'], values = [n_recetas_activas, n_recetas_inactivas])
                    donut_recetas = plotly_to_html(donut_recetas)

                    donut_elaboraciones = create_donut_chart('', 'ELABORACIONES', colores = ['green', 'red'], labels = ['ACTIVO', 'NO ACTIVO'], values = [n_elaboraciones_activas, n_elaboraciones_inactivas])
                    donut_elaboraciones = plotly_to_html(donut_elaboraciones)

                    donut_n_fichas_tecnicas = create_donut_chart('', 'FICHAS TÉCNICAS', colores = ['green', 'red'], labels = ['ACTIVO', 'NO ACTIVO'], values = [n_fichas_tecnicas_activas, n_fichas_tecnicas_inactivas])
                    donut_n_fichas_tecnicas = plotly_to_html(donut_n_fichas_tecnicas)

                    donut_n_etiquetas = create_donut_chart('', 'ETIQUETAS', colores = ['blue', 'red'], labels = ['FAVORITA', 'NO FAVORITA'], values = [n_etiquetas_favoritas, n_etiquetas_nofavoritas])
                    donut_n_etiquetas = plotly_to_html(donut_n_etiquetas)



                    items.append(
                        ui.accordion_panel(
                            f"{modulo}".upper(),
                            ui.div(
                                ui.p(content),
                                ui.div(
                                    ui.HTML(tarjeta_recetas),
                                    ui.HTML(tarjeta_fichas_tecnicas),
                                    ui.HTML(tarjeta_elaboraciones),
                                    ui.HTML(tarjeta_etiquetas),
                                    style="display: flex; flex-wrap: wrap; gap: 20px; align-items: center;"  # Alinea los elementos horizontalmente y agrega espacio entre ellos
                                ),
                                ui.div(
                                    # ui.HTML(fig_circular_recetas.to_html(full_html=False)),
                                    ui.HTML(donut_recetas),
                                    ui.HTML(donut_elaboraciones),
                                    ui.HTML(donut_n_fichas_tecnicas),
                                    ui.HTML(donut_n_etiquetas),
                                    style="display: flex; flex-wrap: wrap; gap: 20px; align-items: center;"  # Alinea los elementos horizontalmente y agrega espacio entre ellos
                                )
                            ),
                            class_="accordion-panel-enabled"
                        )
                    )
                    end_cocina = time.time()
                    time_cocina = end_cocina - start_cocina
                    print("T-COCINA:", time_cocina)


# ---------------------------------------------------------------------------------------------------------

                elif modulo == 'almacenes':
                    start_almacenes = time.time()
                    modulo_almacenes_df1, modulo_almacenes_df2, modulo_almacenes_df3 = obtener_datos_almacen(cliente_data['id'].iloc[0], fecha_inicio_rango, fecha_fin)

                    # Cálculos
                    num_almacenes = modulo_almacenes_df1["N_ALMACENES"].sum()
                    num_inventarios = modulo_almacenes_df1["N_INVENTARIOS"].sum()
                    num_mermas = modulo_almacenes_df1["N_MERMAS"].sum()
                    num_inventarios_abiertos = modulo_almacenes_df1["N_INVENTARIOS_ABIERTOS"].sum()
                    num_inventarios_cerrados = num_inventarios - num_inventarios_abiertos
                    num_traslados = modulo_almacenes_df2["N_TRASLADOS"].sum()
                    traslados_mismo_local, traslados_dif_local = contar_traslados(modulo_almacenes_df2)

                    # Generar la tabla HTML
                    roles = modulo_almacenes_df1
                    tabla_html = roles.to_html(index=False, classes='table table-striped')

                    # CSS para ajustar el ancho de las columnas de la tabla
                    tabla_css = """
                    <style>
                    .table {
                        width: auto;
                        table-layout: auto;
                        margin: auto;
                    }
                    .table th, .table td {
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        max-width: 150px;
                    }
                    </style>
                    """

                    # Crear gráficos de donut
                    colores_inventarios = ['#1f77b4', '#ff7f0e']  # Colores para inventarios
                    fig_inventarios = create_donut_chart(
                        chart_title='Inventarios Abiertos vs Cerrados',
                        center_text=f'Total\n{num_inventarios}',
                        colores=colores_inventarios,
                        labels=['Inventarios Abiertos', 'Inventarios Cerrados'],
                        values=[num_inventarios_abiertos, num_inventarios_cerrados]
                    )
                    fig_inventarios_html = plotly_to_html(fig_inventarios)

                    colores_traslados = ['#2ca02c', '#d62728']  # Colores para traslados
                    fig_traslados = create_donut_chart(
                        chart_title='Traslados: Mismo Local vs Diferentes Locales',
                        center_text=f'Total\n{num_traslados}',
                        colores=colores_traslados,
                        labels=['Mismo Local', 'Diferentes Locales'],
                        values=[traslados_mismo_local, traslados_dif_local]
                    )
                    fig_traslados_html = plotly_to_html(fig_traslados)

                    # Mapa de calor
                    heatmap_almacen_1 = heatmap_almacen(modulo_almacenes_df2)
                    heatmap_almacen_1_html = plotly_to_html(heatmap_almacen_1)

                    # Gráfico de sectores
                    fig_sectores = crear_grafico_sectores_almacenes(modulo_almacenes_df3, labels_column='nombre', titulo='Distribución de Tipos de Almacenes', values_column='N_ALMACEN_TIPO')
                    fig_sectores_html = plotly_to_html(fig_sectores)

                    # Tarjetas informativas
                    tarjeta_num_inventarios = f"""
                    <div style="background-color: #B0C4DE; padding: 20px; border-radius: 10px; text-align: center; flex: 1;">
                        <h4>Número de Inventarios</h4>
                        <p>{num_inventarios}</p>
                    </div>
                    """

                    tarjeta_num_almacenes = f"""
                    <div style="background-color: #B0C4DE; padding: 20px; border-radius: 10px; text-align: center; flex: 1;">
                        <h4>Número de Almacenes</h4>
                        <p>{num_almacenes}</p>
                    </div>
                    """

                    tarjeta_num_mermas = f"""
                    <div style="background-color: #B0C4DE; padding: 20px; border-radius: 10px; text-align: center; width: 100%;">
                        <h4>Número de Mermas</h4>
                        <p>{num_mermas}</p>
                    </div>
                    """

                    # Tarjeta informativa para el número de traslados
                    tarjeta_num_traslados = f"""
                    <div style="background-color: #B0C4DE; padding: 20px; border-radius: 10px; text-align: center;">
                        <h4>Número de Traslados</h4>
                        <p>{num_traslados}</p>
                    </div>
                    """

                    # Tarjeta superior con la tabla a la izquierda y dos gráficos de donut + tarjetas a la derecha
                    tarjeta_superior = f"""
                    <div style="background-color: #E0F7FA; padding: 20px; border-radius: 10px; display: flex; gap: 20px; align-items: stretch;">
                        <div style="flex: 1;">
                            {tabla_css + tabla_html}
                        </div>
                        <div style="flex: 1; display: flex; flex-direction: column; gap: 20px;">
                            <div style="display: flex; gap: 20px;">
                                <div style="flex: 1;">
                                    {fig_inventarios_html}
                                </div>
                                <div style="flex: 1;">
                                    {fig_sectores_html}
                                </div>
                            </div>
                            <div style="display: flex; gap: 20px;">
                                {tarjeta_num_inventarios}
                                {tarjeta_num_almacenes}
                            </div>
                            <div>
                                {tarjeta_num_mermas}
                            </div>
                        </div>
                    </div>
                    """

                    # Tarjeta inferior con la tarjeta de traslados y heatmap
                    tarjeta_inferior = f"""
                        <div style="background-color: #E0F7FA; padding: 20px; border-radius: 10px; display: flex; gap: 20px;">
                            <div style="flex: 0.4; display: flex; flex-direction: column; gap: 20px; align-items: center; justify-content: center;">
                                {tarjeta_num_traslados}
                                {fig_traslados_html}
                            </div>
                            <div style="flex: 0.6;">
                                {heatmap_almacen_1_html}
                            </div>
                        </div>
                    """

                    # Añadir las tarjetas al panel de acordeón
                    items.append(
                        ui.accordion_panel(
                            f"{modulo}".upper(),
                            ui.div(
                                ui.p(content),
                                ui.div(
                                    ui.HTML(tarjeta_superior),
                                    style="width: 100%; margin-bottom: 20px;"  # Ocupa todo el ancho y añade espacio debajo
                                ),
                                ui.div(
                                    ui.HTML(tarjeta_inferior),
                                    style="width: 100%;"  # Ocupa todo el ancho
                                )
                            ),
                            class_="accordion-panel-enabled"
                        )
                    )
                    end_almacenes = time.time()
                    time_almacenes = end_almacenes - start_almacenes
                    print("T-ALMACENES:", time_almacenes)


# ----------------------------------------------------------------------------------------------------------------------------------------


                elif modulo == 'dashboards':
                    start_dash = time.time()

                    modulo_dash_permisos = obtener_datos_dashboards(cliente_data['id'].iloc[0])

                    # Generar tarjetas HTML para los dashboards
                    def generar_tarjetas_dashboards(modulo_dash_permisos):
                        tarjetas_html = ""
                        for key, value in modulo_dash_permisos.get('backoffice', {}).items():
                            # Asignar color de fondo y borde según el valor
                            color_fondo = "#D1FAE5" if value else "#FEE2E2"
                            color_borde = "#10B981" if value else "#EF4444"  # Verde si es true, rojo si es false

                            if key == "dashboard":
                                nombre = "PANEL DE INICIO"
                                tarjeta_class = "tarjeta-dashboard"
                            elif key == "dashboard_productos_ventas":
                                nombre = "PANEL DE PRODUCTOS DE VENTA"
                                tarjeta_class = "tarjeta-dashboard"
                            elif key == "dashboard_control_productos":
                                nombre = "PANEL DE CONTROL DE PRODUCTOS"
                                tarjeta_class = "tarjeta-dashboard"
                            elif key == "dashboard_checklist_apcc_auditorias_plataforma":
                                nombre = "PANEL DE CHECKLISTS, APCC, AUDITORÍAS Y PLATAFORMA"
                                tarjeta_class = "tarjeta-dashboard-large"  # Clase especial para tarjeta más ancha
                            elif key == "dashboards_dinamicos":
                                nombre = "PANELES DINÁMICOS"
                                tarjeta_class = "tarjeta-dashboard"
                            else:
                                nombre = f"PANEL DE {key.replace('_', ' ').replace('dashboard', '').upper()}"
                                tarjeta_class = "tarjeta-dashboard"

                            tarjeta = f"""
                            <div class="{tarjeta_class}" style="background-color: {color_fondo}; border: 1px solid {color_borde};">
                                <h4>{nombre}</h4>
                            </div>
                            """
                            tarjetas_html += tarjeta
                        return tarjetas_html

                    # HTML para mostrar las tarjetas
                    tarjetas_html = generar_tarjetas_dashboards(modulo_dash_permisos)

                    items.append(
                        ui.accordion_panel(
                            f"{modulo}".upper(),
                            ui.div(
                                ui.p(content),
                                ui.div(
                                    ui.HTML(f"""
                                    <style>
                                        .tarjeta-dashboard {{
                                            padding: 20px;
                                            border-radius: 10px;
                                            text-align: center;
                                            transition: transform 0.2s, box-shadow 0.2s;
                                            grid-column: span 1;  /* Normal span */
                                        }}
                                        .tarjeta-dashboard-large {{
                                            padding: 20px;
                                            border-radius: 10px;
                                            text-align: center;
                                            transition: transform 0.2s, box-shadow 0.2s;
                                            grid-column: span 2;  /* Ocupa dos columnas */
                                        }}
                                        .tarjeta-dashboard:hover, .tarjeta-dashboard-large:hover {{
                                            transform: scale(1.05);
                                            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                                        }}
                                        .tarjeta-dashboard h4, .tarjeta-dashboard-large h4 {{
                                            font-size: 18px;
                                            margin: 0;
                                            font-weight: bold;
                                        }}
                                    </style>
                                    <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 20px; align-items: center;">
                                        {tarjetas_html}
                                    </div>
                                    """),
                                    style="width: 100%;"  # Asegura que el contenedor se ajuste al ancho disponible
                                )
                            ),
                            class_="accordion-panel-enabled"
                        )
                    )
                    end_dash = time.time()
                    time_dash = end_dash - start_dash
                    print("T-DASH:", time_dash)
















# -------------------------------------------------------------------------------------------------------------------------------------------------------


                elif modulo == 'finanzas':
                    start_finanzas = time.time()
                    # Obtener los datos de finanzas
                    #modulo_finanzas_df1, modulo_finanzas_df2, modulo_finanzas_df12, modulo_finanzas_df3, recetas_inactivas_df = obtener_datos_finanzas_1(cliente_data['id'].iloc[0])
                    # DF 1 Y DF 2 HAY QUE BORRARLOS, EL DF 12 ES EL BUENO
                    modulo_finanzas_df12, modulo_finanzas_df3 = obtener_datos_finanzas_1(cliente_data['id'].iloc[0], fecha_inicio_rango, fecha_fin)
                    
                    # INGRESOS
                    total_ingresos = modulo_finanzas_df3['IMPORTE_TOTAL'].sum()
                    total_cierres_caja = modulo_finanzas_df3['N_CIERRES'].sum()

                    df_turnos = modulo_finanzas_df3.replace({'nombre_turno': ''}, pd.NA).fillna({'nombre_turno': 'Sin Turno Asignado'}).groupby('nombre_turno', as_index=False).agg({'IMPORTE_TOTAL': 'sum', 'N_CIERRES': 'sum'})
                    df_turnos['ingreso_medio_por_cierre'] = df_turnos['IMPORTE_TOTAL'] / df_turnos['N_CIERRES']

                    def obtener_importe_medio(df, nombre_turno):
                        if nombre_turno in df['nombre_turno'].values:
                            return round(df.loc[df['nombre_turno'] == nombre_turno, 'ingreso_medio_por_cierre'].values[0], 2)
                        else:
                            return '-'
                    
                    importe_medio_cierre_desayuno = obtener_importe_medio(df_turnos, 'Desayuno')
                    importe_medio_cierre_comida = obtener_importe_medio(df_turnos, 'Comida')
                    importe_medio_cierre_cena = obtener_importe_medio(df_turnos, 'Cena')

                    print("eeey", importe_medio_cierre_desayuno, importe_medio_cierre_comida, importe_medio_cierre_cena )

                    #df_importes_turno = df_turnos.to_html(index=False, classes='table table-striped')
                    #df_cierres_turno = df_turnos.to_html(index=False, classes='table table-striped')

                    color_map = {
                        'Sin Turno Asignado': '#000000',  # Negro
                        'Desayuno': '#821DB8',    # Amarillo
                        'Comida': '#FFB200',    # Morado
                        'Cena': '#FF5733'     # Verde
                    }

                    # Gráfico de sectores DISTRIBUCIÓN DE CIERRES DE CAJA POR TURNO
                    plot_ingresos_cierres_turno = crear_grafico_pastel_finanzas(df_turnos, columna_niveles='nombre_turno', columna_valores='N_CIERRES', 
                     titulo='CIERRES POR TURNO', ancho_grafico=450, altura_grafico=400, color_map=color_map, nivel_negro= 'Sin Turno Asignado')
                    plot_ingresos_cierres_turno = plotly_to_html(plot_ingresos_cierres_turno)

                    # Gráfico de sectores APORTACIÓN DE INGRESOS POR TURNO
                    plot_ingresos_aportacion_turno = crear_grafico_pastel_finanzas(df_turnos, columna_niveles='nombre_turno', columna_valores='IMPORTE_TOTAL', 
                     titulo='APORTACIÓN DE INGRESOS POR TURNO', ancho_grafico=450, altura_grafico=400, color_map=color_map, nivel_negro= 'Sin Turno Asignado')
                    plot_ingresos_aportacion_turno = plotly_to_html(plot_ingresos_aportacion_turno)

                    # Gráfico de BARRAS IMPORTE MEDIO POR TURNO
                    plot_ingresos_importe_turno = plot_barras_ingresos([importe_medio_cierre_desayuno, importe_medio_cierre_comida, importe_medio_cierre_cena],
                                          niveles = ['Desayuno', 'Comida', 'Cena'],
                                          colores = ['#821DB8','#FFB200','#FF5733'], 
                                          titulo='IMPORTE MEDIO POR TURNO', 
                                          ancho_grafico=800, altura_grafico=500)
                    
                    plot_ingresos_importe_turno =  plotly_to_html(plot_ingresos_importe_turno)
                                          

                    #  GASTOS

                    
                    g_sectores_gastos_generales = g_sectores_agrupado_gastos(df = modulo_finanzas_df12, columna_niveles = 'nombre_familia_gasto', columna_valores = 'N_GASTOS', titulo = 'Distribución Gastos Generales', ancho_grafico = 650, altura_grafico = 450)
                    g_sectores_gastos_generales = plotly_to_html(g_sectores_gastos_generales)

                    df_personal = modulo_finanzas_df12.loc[modulo_finanzas_df12['ASOCIADO_A_PERSONAL'] == 'SI']
                    g_sectores_gastos_personales = g_sectores_agrupado_gastos(df = df_personal, columna_niveles = 'nombre_familia_gasto', columna_valores = 'N_GASTOS', titulo = 'Distribución Gastos de Personal', ancho_grafico = 650, altura_grafico = 450)
                    g_sectores_gastos_personales = plotly_to_html(g_sectores_gastos_personales)

                    total_gastos_generales_registrados = modulo_finanzas_df12['N_GASTOS'].sum()
                    total_gastos_personal_registrados = modulo_finanzas_df12.loc[modulo_finanzas_df12['ASOCIADO_A_PERSONAL'] == 'SI', 'N_GASTOS'].sum()




                    # BORRAR ESTO DE ABAJO

                    # Resumir los datos
                    #total_gastos_generales = modulo_finanzas_df1['IMP_GASTOS_PERSONAL'].sum()
                    #total_gastos_personal = modulo_finanzas_df2['IMPORTE_TOTAL'].sum()
                    

                    # Crear DataFrame combinado para gráfico de barras apiladas
                    #df_gastos_generales = modulo_finanzas_df1.groupby('id_local')['IMP_GASTOS_PERSONAL'].sum().reset_index()
                    #df_gastos_personal = modulo_finanzas_df2.groupby('id_local')['IMPORTE_TOTAL'].sum().reset_index()
                    #df_ingresos = modulo_finanzas_df3.groupby('id_local')['IMPORTE_TOTAL'].sum().reset_index()

                    # Obtener la lista completa de todos los 'id_local'
                    #all_id_local = pd.concat([df_gastos_generales['id_local'], df_gastos_personal['id_local'], df_ingresos['id_local']]).unique()

                    # Crear un DataFrame completo con todos los 'id_local' y valores inicializados a 0
                    #df_completo = pd.DataFrame({'id_local': all_id_local})
                    #df_completo = pd.merge(df_completo, df_gastos_generales, on='id_local', how='left').fillna(0)
                    #df_completo = pd.merge(df_completo, df_gastos_personal, on='id_local', how='left').fillna(0)
                    #df_completo = pd.merge(df_completo, df_ingresos, on='id_local', how='left').fillna(0)

                    # Renombrar columnas
                    #df_completo = df_completo.rename(columns={'IMP_GASTOS_PERSONAL': 'Gastos Generales', 'IMPORTE_TOTAL_x': 'Gastos Personal', 'IMPORTE_TOTAL_y': 'Ingresos'})

                    # Gráfico de barras apiladas: Ingresos y gastos por local
                    #fig_barras_apiladas = px.bar(
                    #    df_completo, 
                    #    x='id_local', 
                    #    y=['Ingresos', 'Gastos Generales', 'Gastos Personal'], 
                    #    title='Ingresos y Gastos por Local', 
                    #    labels={'value': 'Importe', 'variable': 'Tipo'},
                    #    color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
                    #)

                    # Gráfico circular: Desglose de gastos generales por tipo de gasto
                    # fig_circular_gastos_generales = px.pie(
                    #     modulo_finanzas_df1, 
                    #     values='IMP_GASTOS_PERSONAL', 
                    #     names='nombre_familia_gasto', 
                    #     title='Desglose de Gastos Generales por Tipo de Gasto',
                    #     color_discrete_sequence=px.colors.sequential.RdBu
                    # )

                    # Gráfico circular: Desglose de gastos de personal por tipo de gasto
                    # fig_circular_gastos_personal = px.pie(
                    #     modulo_finanzas_df2, 
                    #     values='IMPORTE_TOTAL', 
                    #     names='nombre_familia', 
                    #     title='Desglose de Gastos de Personal por Tipo de Gasto',
                    #     color_discrete_sequence=px.colors.sequential.RdBu
                    # )

                    # Verificar si la columna 'nombre_turno' existe en modulo_finanzas_df3 antes de usarla
                    # if 'nombre_turno' in modulo_finanzas_df3.columns:
                    #     # Tabla: Resumen de ingresos por turno
                    #     tabla_ingresos_turno = modulo_finanzas_df3[['nombre_turno', 'IMPORTE_TOTAL', 'N_CIERRES']]
                    #     tabla_html_ingresos_turno = tabla_ingresos_turno.to_html(index=False, classes='table table-striped')
                        
                    #     # CSS para ajustar el ancho de las columnas de la tabla
                    #     tabla_css_ingresos_turno = """
                    #     <style>
                    #     .table {
                    #         width: auto;
                    #         table-layout: auto;
                    #         margin: auto;
                    #     }
                    #     .table th, .table td {
                    #         white-space: nowrap;
                    #         overflow: hidden;
                    #         text-overflow: ellipsis;
                    #         max-width: 150px;
                    #     }
                    #     </style>
                    #     """
                    # else:
                    #     tabla_html_ingresos_turno = "<p>Los datos de 'nombre_turno' no están disponibles.</p>"
                    #     tabla_css_ingresos_turno = ""

                    
                    fondo_verde_claro = "#E6FFE6"

                    tarjeta_izquierda = f"""
                        <div style="background-color: {fondo_verde_claro}; padding: 20px; border-radius: 10px; flex: 1; display: flex; flex-direction: column; gap: 20px;">
                            <h3 style="font-weight: bold; text-align: center; color: black;">RESUMEN DE GASTOS</h3>
                            
                            <!-- Gráfico HTML fig_1 -->
                            <div style="display: flex; justify-content: center;">
                                {g_sectores_gastos_generales}
                            </div>
                            
                            <!-- Tarjeta negra 'G' -->
                            <div style="display: flex; gap: 10px; margin-top: 20px;">
                                <div style="flex: 1; padding: 10px; background-color: #000000; border-radius: 5px; height: 100px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                    <h5 style="font-weight: bold; text-align: center; color: white;">NÚMERO DE GASTOS</h5>
                                    <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{total_gastos_generales_registrados}</p>
                                </div>
                            </div>
                            
                            <!-- Gráfico HTML fig_2 -->
                            <div style="display: flex; justify-content: center;">
                                {g_sectores_gastos_personales}
                            </div>
                            
                            <!-- Tarjeta negra 'P' -->
                            <div style="display: flex; gap: 10px; margin-top: 20px;">
                                <div style="flex: 1; padding: 10px; background-color: #000000; border-radius: 5px; height: 100px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                    <h5 style="font-weight: bold; text-align: center; color: white;">NÚMERO DE GASTOS DE PERSONAL</h5>
                                    <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{total_gastos_personal_registrados}</p>
                                </div>
                            </div>
                        </div>
                    """


                    tarjeta_derecha = f"""
                        <div style="background-color: {fondo_verde_claro}; padding: 20px; border-radius: 10px; flex: 1; display: flex; flex-direction: column; gap: 20px;">
                            <h3 style="font-weight: bold; text-align: center; color: black;">INGRESOS</h3>
                            
                            <div style="display: flex; justify-content: center;">
                                {plot_ingresos_cierres_turno}
                                {plot_ingresos_aportacion_turno}
                            </div>
                            <div style="display: flex; gap: 10px;">
                                        <div style="flex: 1; padding: 10px; background-color: #000000; border-radius: 5px; height: 100px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                            <h5 style="font-weight: bold; text-align: center; color: white;">NÚMERO DE CIERRES DE CAJA</h5>
                                            <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{total_cierres_caja}</p>
                                        </div>
                            </div>
                            <div style="display: flex; justify-content: center;">
                                {plot_ingresos_importe_turno}
                            </div>
                            
                            <!-- Tarjetas en la parte inferior -->
                            <div style="display: flex; gap: 10px; margin-top: 20px; justify-content: space-around;">
                                <div style="flex: 1; padding: 10px; background-color: #821DB8; border-radius: 5px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                    <h5 style="font-weight: bold; text-align: center; color: white;">IMPORTE MEDIO DESAYUNO</h5>
                                    <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{importe_medio_cierre_desayuno} €</p>
                                </div>
                                <div style="flex: 1; padding: 10px; background-color: #FFB200; border-radius: 5px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                    <h5 style="font-weight: bold; text-align: center; color: white;">IMPORTE MEDIO COMIDA</h5>
                                    <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{importe_medio_cierre_comida} €</p>
                                </div>
                                <div style="flex: 1; padding: 10px; background-color: #FF5733; border-radius: 5px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                    <h5 style="font-weight: bold; text-align: center; color: white;">IMPORTE MEDIO CENA</h5>
                                    <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{importe_medio_cierre_cena} €</p>
                                </div>
                            </div>
                        </div>
                    """

                    # Añadir las tarjetas al panel de acordeón, colocando las tarjetas una al lado de la otra
                    items.append(
                        ui.accordion_panel(
                            f"{modulo}".upper(),
                            ui.div(
                                ui.p(content),
                                ui.div(
                                    ui.div(ui.HTML(tarjeta_izquierda), style="flex: 5.5;"),  # Tarjeta izquierda ajustada con flex
                                    ui.div(ui.HTML(tarjeta_derecha), style="flex: 4.5;"),   # Tarjeta derecha ajustada con flex
                                    style="display: flex; width: 100%; gap: 20px;"  # Colocar las tarjetas en una fila con espacio entre ellas
                                ),
                                style="width: 100%; margin-bottom: 20px;"  # Ocupa todo el ancho y añade espacio debajo
                            ),
                            class_="accordion-panel-enabled"
                        )
                    )
                    end_finanzas = time.time()
                    time_finanzas = end_finanzas - start_finanzas
                    print("T-FINANZAS:", time_finanzas)







# ------------------------------------------------------------------------------------------------------------------------------



                elif modulo == 'ocr':
                    start_ocr= time.time()
                    # Obtener los datos de finanzas
                    ocr_df1, ocr_df2, ocr_df3 = obtener_datos_ocr(cliente_data['id'].iloc[0], fecha_inicio_rango, fecha_fin)
                    
                    total_albaranes_ocr = ocr_df1['N_ALBARANES_OCR'][0]
                    total_modelos_ocr = ocr_df1['N_MODELOS_OCR'][0]
                    total_albaranes_correctos = ocr_df1['N_ALBARANES_DIRECTOS'][0]
                    total_fallos_tabla_items = ocr_df1['F_TABLA_ITEMS'][0]
                    total_fallos_encabezado = ocr_df1['F_ENCABEZADO'][0]


                    total_fallos_ocr = total_albaranes_ocr - total_albaranes_correctos
                    total_correctos_ocr_tabla = total_albaranes_ocr - total_fallos_tabla_items
                    total_correctos_ocr_encabezado = total_albaranes_ocr - total_fallos_encabezado

                    tabla_modelos = filtro_ultima_version(ocr_df2, 'id_model').head(15).to_html(index=False, classes='table table-striped')
                    tabla_incidencias = procesar_incidencias(ocr_df3)
                    tabla_incidencias_html = procesar_incidencias(ocr_df3).to_html(index=False, classes='table table-striped')

                    
                    gbarras_incidencias_ocr = plot_barras_compras(tabla_incidencias,label_column='incidencia',
                                                value_columns = ['TOTAL'],
                                                colors = ['black'],
                                                legend_names = ['A'],
                                                graph_title = 'Incidencia',
                                                xaxis_title = 'Número de Ocurrencias',
                                                yaxis_title = 'Locales',
                                                total_accumulated=False
                                            )
                    gbarras_incidencias_ocr = plotly_to_html(gbarras_incidencias_ocr)


                    donut_fallos_ocr = create_donut_chart(
                        chart_title='OCR',
                        center_text=f'ALBARANES',
                        colores=['green', 'red'],
                        labels=['CORRECTOS', 'INCORRECTOS'],
                        values=[total_albaranes_correctos, total_fallos_ocr]
                    )

                    donut_fallos_ocr = plotly_to_html(donut_fallos_ocr)

                    donut_fallos_tabla_ocr = create_donut_chart_ocr(
                        chart_title='OCR TABLA',
                        center_text=f'ALBARANES',
                        colores=['orange', 'red'],
                        labels=['CORRECTOS', 'INCORRECTOS'],
                        values=[total_correctos_ocr_tabla, total_fallos_tabla_items]
                    )

                    donut_fallos_tabla_ocr = plotly_to_html(donut_fallos_tabla_ocr)

                    donut_fallos_encabezado_ocr = create_donut_chart_ocr(
                        chart_title='OCR ENCABEZADO',
                        center_text=f'ALBARANES',
                        colores=['blue', 'red'],
                        labels=['CORRECTOS', 'INCORRECTOS'],
                        values=[total_correctos_ocr_encabezado, total_fallos_encabezado]
                    )

                    donut_fallos_encabezado_ocr = plotly_to_html(donut_fallos_encabezado_ocr)

                    fondo_marron_claro = "#F5C4AD"

                    tarjeta_izquierda = f"""
                        <div style="background-color: {fondo_marron_claro}; padding: 20px; border-radius: 10px; flex: 1; display: flex; flex-direction: column; gap: 20px;">
                            <h3 style="font-weight: bold; text-align: center; color: black;">MODELOS OCR</h3>
                            
                            <!-- Tabla 1 -->
                            <div style="display: flex; justify-content: center;">
                                <div style="width: auto;">
                                    {tabla_modelos}
                                </div>
                            </div>

                            
                            <!-- Total 1 -->
                            <div style="display: flex; justify-content: center; margin-top: 20px;">
                                <div style="flex: 1; padding: 10px; background-color: #000000; border-radius: 5px; height: 100px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                    <h5 style="font-weight: bold; text-align: center; color: white;">MODELOS OCR UTILIZADOS</h5>
                                    <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{total_modelos_ocr}</p>
                                </div>
                            </div>
                        </div>
                    """


                    tarjeta_derecha = f"""
                        <div style="background-color: {fondo_marron_claro}; padding: 20px; border-radius: 10px; display: flex; flex-direction: column; gap: 20px;">
                            <h3 style="font-weight: bold; text-align: center; color: black;">ALBARANES DIGITALIZADOS</h3>
                            
                            <div style="display: flex; gap: 20px; align-items: stretch;">
                                <!-- Columna izquierda (60% del ancho) -->
                                <div style="flex: 65; display: flex; flex-direction: column; gap: 20px; align-items: center; justify-content: center;">
                                    <!-- Div que contiene el donut y gráfico de barras -->
                                    <div style="display: flex; flex-direction: column; gap: 20px; align-items: center;">
                                        {donut_fallos_ocr}
                                        {gbarras_incidencias_ocr}
                                    </div>
                                </div>
                                
                                <!-- Columna derecha (40% del ancho) -->
                                <div style="flex: 35; display: flex; flex-direction: column; gap: 20px; align-items: center; justify-content: center;">
                                <!-- Div que contiene el donut de fallos en el encabezado y en la tabla -->
                                <div style="display: flex; flex-direction: column; gap: 20px; align-items: center;">
                                    {donut_fallos_encabezado_ocr}
                                    {donut_fallos_tabla_ocr}
                                </div>
                            </div>
                            </div>

                            <!-- Sección de tarjetas inferior alineada -->
                            <div style="display: flex; gap: 20px;">
                                <!-- Tarjeta izquierda: Número total de albaranes -->
                                <div style="flex: 65;">
                                    <div style="padding: 10px; background-color: #FF5733; border-radius: 5px; height: 100px; display: flex; align-items: center; justify-content: center; flex-direction: column; box-sizing: border-box;">
                                        <h5 style="font-weight: bold; text-align: center; color: white;">NÚMERO TOTAL DE ALBARANES</h5>
                                        <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{total_albaranes_ocr}</p>
                                    </div>
                                </div>

                                <!-- Tarjeta derecha: Total Fallo en el encabezado -->
                                <div style="flex: 35;">
                                    <div style="padding: 10px; background-color: #FFB200; border-radius: 5px; height: 100px; display: flex; align-items: center; justify-content: center; flex-direction: column; box-sizing: border-box;">
                                        <h5 style="font-weight: bold; text-align: center; color: white;">TOTAL FALLO EN EL ENCABEZADO</h5>
                                        <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{total_fallos_encabezado}</p>
                                    </div>
                                </div>
                            </div>

                            <div style="display: flex; gap: 20px; margin-top: 20px;">
                                <!-- Tarjeta izquierda: Total albaranes correctos -->
                                <div style="flex: 65;">
                                    <div style="padding: 10px; background-color: #000000; border-radius: 5px; height: 100px; display: flex; align-items: center; justify-content: center; flex-direction: column; box-sizing: border-box;">
                                        <h5 style="font-weight: bold; text-align: center; color: white;">TOTAL ALBARANES CORRECTOS</h5>
                                        <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{total_albaranes_correctos}</p>
                                    </div>
                                </div>

                                <!-- Tarjeta derecha: Total Fallos en la tabla de productos -->
                                <div style="flex: 35;">
                                    <div style="padding: 10px; background-color: #821DB8; border-radius: 5px; height: 100px; display: flex; align-items: center; justify-content: center; flex-direction: column; box-sizing: border-box;">
                                        <h5 style="font-weight: bold; text-align: center; color: white;">TOTAL FALLOS EN LA TABLA DE PRODUCTOS</h5>
                                        <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{total_fallos_tabla_items}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        """



                    # Añadir las tarjetas al panel de acordeón, colocando las tarjetas una al lado de la otra
                    items.append(
                        ui.accordion_panel(
                            f"{modulo}".upper(),
                            ui.div(
                                ui.p(content),
                                ui.div(
                                    ui.div(ui.HTML(tarjeta_izquierda), style="flex: 3;"),  # Tarjeta izquierda ajustada con flex (35%)
                                    ui.div(ui.HTML(tarjeta_derecha), style="flex: 7;"),   # Tarjeta derecha ajustada con flex (65%)
                                    style="display: flex; width: 100%; gap: 20px;"  # Colocar las tarjetas en una fila con espacio entre ellas
                                ),
                                style="width: 100%; margin-bottom: 20px;"  # Ocupa todo el ancho y añade espacio debajo
                            ),
                            class_="accordion-panel-enabled"
                        )
                    )
                    end_ocr = time.time()
                    time_ocr = end_ocr - start_ocr
                    print("T-OCR:", time_ocr)















# ---------------------------------------------------------------------------------------------------------------------------

                elif modulo == 'tareas':
                    start_tareas = time.time()
                    modulo_checklists_df1, modulo_checklists_df2, modulo_checklists_df3, modulo_checklists_df4 = obtener_datos_checklists(cliente_data['id'].iloc[0], fecha_inicio_rango, fecha_fin)
    
                    # Obtenemos los datos
                    df_checklists = modulo_checklists_df1
                    df2_checklists = modulo_checklists_df2
                    df3_checklists = modulo_checklists_df3
                    df4_checklists = modulo_checklists_df4

                    # Plantillas generadas
                    total_modelos = df_checklists['N_MODELOS'].sum()
                    modelos_activos = df_checklists['N_MODELOS_ACTIVOS'].sum()
                    modelos_activos_auditorias = df_checklists['N_MODELOS_ACTIVOS_AUDITORIAS'].sum()
                    modelos_activos_valorables = df_checklists['N_MODELOS_ACTIVOS_VALORABLES'].sum()

                    # Tareas
                    total_tareas = df_checklists['N_TAREAS'].sum()
                    tareas_activas = df_checklists['N_TAREAS_ACTIVAS'].sum()
                    tareas_cerradas = df_checklists['N_TAREAS_ACTIVAS_CERRADAS'].sum()
                    tareas_unicas = df2_checklists['N_TAREAS'][0]

                    # Asignaciones
                    total_asignaciones = df_checklists['N_ASIG'].sum()
                    total_asignaciones_activas = df_checklists['N_ASIG_ACTIVAS'].sum()
                    try:
                        prevision_dias_asignaciones = df2_checklists['SUMA_PREVISION_DIAS'][0] / df2_checklists['SUMA_PREVISION_DIAS'][0]
                    except (ZeroDivisionError, IndexError):
                        prevision_dias_asignaciones = 0
                    
                    # Recurrencias                     
                    total_recurrencias = df_checklists['N_RECURRENCIAS'].sum()
                    total_recurrencias_activas = df_checklists['N_RECURRENCIAS_ACTIVAS'].sum()
                    tabla_frecuencias_recurrencias = modulo_checklists_df3




                    
                    # Crear el gráfico de anillos
                    anillos_tareas = create_anillos_tareas(total_tareas, tareas_activas, tareas_cerradas)
                    anillos_tareas_html = plotly_to_html(anillos_tareas)

                    # Crear el gráfico de barras CHECLISTS (TAREAS)
                    g_barras_tareas = crear_grafico_tareas(
                        df=df_checklists,
                        y_var='nombre_local',
                        tareas_activas='N_TAREAS_ACTIVAS',
                        tareas_cerradas='N_TAREAS_ACTIVAS_CERRADAS',
                        title='Tareas por Local',
                        labels={'x': 'Cantidad de Tareas', 'y': 'Local'}
                    )
                    g_barras_tareas_html = plotly_to_html(g_barras_tareas)


                    # Crear el gráfico de barras PLANTILLAS
                    g_barras_plantillas = crear_grafico_plantillas(
                        df=df_checklists,
                        y_var='nombre_local',
                        total_modelos='N_MODELOS',
                        modelos_activos='N_MODELOS_ACTIVOS',
                        title='Plantillas Generadas por Local',
                        labels={'x': 'Cantidad de Plantillas', 'y': 'Local'}
                    )
                    g_barras_plantillas_html = plotly_to_html(g_barras_plantillas)

                    # Gráfico de líneas PREVISIÓN DE DÍAS
                    g_lineas_prev_dias, media_prevision_dias = generar_grafico_lineas(df4_checklists)
                    g_lineas_prev_dias_html = plotly_to_html(g_lineas_prev_dias)


                    anillos_asig = create_anillos_asig(total_asignaciones, total_asignaciones_activas, total_recurrencias_activas, df3_checklists)
                    anillos_asig_html = plotly_to_html(anillos_asig)


                    # Gráfico de barras agrupadas: Modelos (activos e inactivos) y recurrencias por local
                    fig_barras_agrupadas = px.bar(
                        df_checklists, 
                        x='nombre_local', 
                        y=['N_MODELOS', 'N_MODELOS_ACTIVOS', 'N_RECURRENCIAS'], 
                        title='Modelos y Recurrencias por Local',
                        labels={'value': 'Cantidad', 'variable': 'Tipo'},
                        barmode='group',
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )

                    # Crear el gráfico de líneas para previsión de días
                    g_lineas = crear_grafico_lineas(df2_checklists, title='Previsión de Días')

                    # Convertir el gráfico a HTML
                    g_lineas_html = plotly_to_html(g_lineas)

                    # Tarjeta para el número de plantillas generadas
                    tarjeta_tareas = f"""
                        <div style="background-color: #FFFACD; padding: 20px; border-radius: 10px; flex-grow: 1; text-align: center;">
                            <h3 style="font-weight: bold;">CHECKLISTS (HOJAS DE VERIFICACIÓN)</h3>
                            <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px;">
                                {g_barras_tareas_html}
                            </div>
                            <div style="display: flex; flex-direction: column; margin-top: 20px; gap: 10px;">
                                <div style="flex: 1; display: flex; gap: 10px;">
                                    <div style="flex: 1; padding: 10px; background-color: #FFD700; border-radius: 5px;">
                                        <h4 style="font-weight: bold;">CHECKLISTS</h4>
                                        <p>{total_tareas}</p>
                                    </div>
                                    <div style="flex: 1; padding: 10px; background-color: #FFD700; border-radius: 5px;">
                                        <h4 style="font-weight: bold;">CAMPOS</h4>
                                        <p>PENDIENTE</p>
                                    </div>
                                    <div style="flex: 1; padding: 10px; background-color: #FFD700; border-radius: 5px;">
                                        <h4 style="font-weight: bold;">CAMPOS POR CHECK</h4>
                                        <p>PENDIENTE</p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        """



                    # <p>Número de Plantillas Generadas: {total_modelos}</p>
                    # Tarjeta para el porcentaje de tareas realizadas
                    porcentaje_tareas_realizadas = 1
                    tarjeta_modelos = f"""
                    <div style="background-color: #FFFACD; padding: 20px; border-radius: 10px; flex-grow: 1; text-align: center;">
                        <h3 style="font-weight: bold;">PLANTILLAS GENERADAS</h3>
                        <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px;">
                            {g_barras_plantillas_html}
                        </div>
                        <div style="flex: 1; display: flex; gap: 10px; margin-top: 20px;">
                                    <div style="flex: 1; padding: 10px; background-color: #90EE90; border-radius: 5px;">
                                        <h4 style="font-weight: bold;">PLANTILLAS</h4>
                                        <p>{total_modelos}</p>
                                    </div>
                                    <div style="flex: 1; padding: 10px; background-color: #90EE90; border-radius: 5px;">
                                        <h4 style="font-weight: bold;">PL. ACTIVAS</h4>
                                        <p>{modelos_activos}</p>
                                    </div>
                                    <div style="flex: 1; padding: 10px; background-color: #90EE90; border-radius: 5px;">
                                        <h4 style="font-weight: bold;">AUDITORÍAS</h4>
                                        <p>{modelos_activos_auditorias}</p>
                                    </div>
                                    <div style="flex: 1; padding: 10px; background-color: #90EE90; border-radius: 5px;">
                                        <h4 style="font-weight: bold;">VALORABLES</h4>
                                        <p>{modelos_activos_valorables}</p>
                                    </div>
                        </div>
                        

                    </div>
                    """



                    # Tarjeta para el total de asignaciones
                    tarjeta_asignaciones = f"""
                    <div style="background-color: #FFFACD; padding: 20px; border-radius: 10px; width: 100%; text-align: center;">
                        <h4 style="font-weight: bold;">ASIGNACIONES Y RECURRENCIAS</h4>
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px;">
                            <!-- Primera mitad: primer gráfico y tarjeta de tareas únicas -->
                            <div style="flex: 1; display: flex; flex-direction: column; align-items: center; gap: 10px;">
                                <div style="background-color: transparent; border-radius: 10px; padding: 20px;">
                                    <!-- Gráfico de anillos -->
                                    {anillos_asig_html}
                                </div>
                                <div style="background-color: #FFD700; border-radius: 5px; padding: 10px; width: 100%;">
                                    <h4 style="font-weight: bold;">ASIGNACIONES</h4>
                                    <p>{total_asignaciones}</p>
                                </div>
                            </div>
                            
                            <!-- Segunda mitad: segundo gráfico y tarjeta de tareas únicas -->
                            <div style="flex: 1; display: flex; flex-direction: column; align-items: center; gap: 10px;">
                                <div style="background-color: transparent; border-radius: 10px; padding: 20px;">
                                    <!-- Gráfico de líneas -->
                                    {g_lineas_prev_dias_html}
                                </div>
                                <div style="background-color: #FFD700; border-radius: 5px; padding: 10px; width: 100%;">
                                    <h4 style="font-weight: bold;">MEDIA PREVISIÓN DÍAS</h4>
                                    <p>{media_prevision_dias}</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    """


                    items.append(
                        ui.accordion_panel(
                            f"{modulo}".upper(),
                            ui.div(
                                ui.p(content),
                                # Primera fila de tarjetas
                                ui.div(
                                    ui.HTML(tarjeta_tareas),
                                    ui.HTML(tarjeta_modelos),
                                    style="display: flex; gap: 20px; align-items: stretch; width: 100%;"
                                ),
                                # Segunda fila de tarjeta (más alargada)
                                ui.div(
                                    ui.HTML(tarjeta_asignaciones),
                                    style="margin-top: 20px; width: 100%;"
                                ),
                                # Gráficos
                                ui.div(
                                    ui.HTML(fig_barras_agrupadas.to_html(full_html=False)),
                                    style="display: flex; flex-wrap: wrap; gap: 20px; align-items: center;"
                                )
                            ),
                            class_="accordion-panel-enabled"
                        )
                    )
                    end_tareas = time.time()
                    time_tareas = end_tareas - start_tareas
                    print("T-TAREAS:", time_tareas)











































# ----------------------------------------------------------------------------------------------------------

                elif modulo == 'compras':
                    start_compras = time.time()
                    df_compras, df2_compras, df3_compras, df5_compras, df67_compras = obtener_datos_compras_2(cliente_data['id'].iloc[0], fecha_inicio_rango, fecha_fin)
                    # Productos
                    num_productos_compra = df_compras['total_productos_compra'][0]
                    num_productos_activos_compra = df_compras['productos_activos_compra'][0]
                    num_productos_inactivos_compra = num_productos_compra - num_productos_activos_compra   
                    productos_sin_proveedor = df_compras['productos_sin_proveedor'][0]
                    productos_con_proveedor = num_productos_compra - productos_sin_proveedor

                    # Albaranes
                    total_albaranes = df3_compras['total_albaranes_generados'].sum()
                    total_albaranes_por_foto = df3_compras['albaranes_por_foto'].sum()
                    total_albaranes_por_cerrar = df3_compras['albaranes_por_cerrar'].sum()

                    # Facturas
                    total_facturas = df3_compras['facturas'].sum()
                    total_facturas_sin_local = df3_compras['facturas_cliente_sin_local'][0]

                    # Pedidos
                    total_pedidos = df3_compras['pedidos_generados'].sum()

                    # Proveedores
                    total_proveedores = df3_compras['total_proveedores'][0]
                    total_proveedores_activos = df3_compras['total_proveedores_activos'][0]
                    total_proveedores_no_activos = total_proveedores - total_proveedores_activos



                    colores_inventarios = ['#FF9999', '#FFFF99']

                    donut_productos_activ = create_donut_chart(
                        chart_title='Productos Activos vs Inactivos',
                        center_text=f'{num_productos_compra} PRODUCTOS',
                        colores=colores_inventarios,
                        labels=['Productos Activos', 'Productos Inactivos'],
                        values=[num_productos_activos_compra, num_productos_inactivos_compra]
                    )

                    donut_productos_prodprov = create_donut_chart(
                        chart_title='Asignación de Proveedor a Producto',
                        center_text=f'{num_productos_compra} PRODUCTOS',
                        colores=colores_inventarios,
                        labels=['Proveedor Asignado', 'Sin Proveedor Asignado'],
                        values=[productos_con_proveedor, productos_sin_proveedor]
                    )

                    donut_productos_activ_html = plotly_to_html(donut_productos_activ)
                    donut_productos_prodprov_html = plotly_to_html(donut_productos_prodprov)

                    fig_sectores_productos = crear_grafico_sectores_almacenes(df2_compras, labels_column='NOMBRE_FAMILIA', titulo='Distribución de Familia Máster de Productos', values_column='N')
                    fig_sectores_productos_html = plotly_to_html(fig_sectores_productos)


                    gbarras_albaranes = plot_barras_compras(df3_compras,'nombre_local',
                                                ['total_albaranes_generados', 'albaranes_por_foto'],
                                                ['black', 'blue'],
                                                ['TOTAL ALBARANES', 'ALBARANES POR FOTO'],
                                                'Distribución de Albaranes por Locales',
                                                'Número de Albaranes',
                                                'Locales',
                                                total_accumulated=True
                                            )
                    
                    gbarras_albaranes_por_cerrar = plot_barras_compras(df3_compras,'nombre_local',
                                                ['albaranes_por_cerrar'],
                                                ['red'],
                                                ['ALBARANES POR CERRAR'],
                                                'Distribución de Albaranes Por Cerrar por Locales',
                                                'Número de Albaranes Por Cerrar',
                                                'Locales',
                                                total_accumulated=True
                                            )
                    

                    gbarras_facturas = plot_barras_compras(df3_compras,'nombre_local',
                                                ['facturas_pagadas', 'facturas_pendiente_pago', 'facturas_sin_estado'],
                                                ['green', 'red', 'grey'],
                                                ['PAGADAS', 'PENDIENTES DE PAGO', 'SIN ESTADO'],
                                                'Distribución de Facturas por Locales',
                                                'Número de Facturas',
                                                'Locales',
                                                total_accumulated=False
                                            )

                    gbarras_albaranes_html = plotly_to_html(gbarras_albaranes)
                    gbarras_albaranes_por_cerrar_html =plotly_to_html(gbarras_albaranes_por_cerrar)
                    gbarras_facturas_html = plotly_to_html(gbarras_facturas)

                    top10_pedidos = df67_compras.sort_values(by='N_PEDIDOS', ascending=False).head(10)
                    tabla_html_top = top10_pedidos.drop(columns=['id_proveedor']).to_html(index=False, classes='table table-striped')

                    fig_sectores_estado_pedidos = crear_grafico_sectores_almacenes(df5_compras, labels_column='ESTADO', titulo='Estado de Pedidos', values_column='N')
                    fig_sectores_estado_pedidos_html = plotly_to_html(fig_sectores_estado_pedidos)

                    donut_proveedores_activ = create_donut_chart(
                        chart_title='Productos Activos vs Inactivos',
                        center_text=f'PROVEEDORES',
                        colores=['green', 'red'],
                        labels=['Proveedores Activos', 'Proveedores Inactivos'],
                        values=[total_proveedores_activos, total_proveedores_no_activos]
                    )

                    donut_proveedores_activ_html = plotly_to_html(donut_proveedores_activ)

                    tabla_estado_pedidos_html = df5_compras.drop(columns=['id_estado']).to_html(index=False, classes='table table-striped')

                    fondo_rojo_claro = "#FFEFEF"  # Un tono de rojo claro

                    # Tarjeta superior con los gráficos
                    tarjeta_superior = f"""
                        <div style="background-color: {fondo_rojo_claro}; padding: 20px; border-radius: 10px; display: flex; flex-direction: column; gap: 20px;">
                            <h3 style="font-weight: bold; text-align: center; color: black;">PRODUCTOS DE COMPRA</h3>
                            <div style="display: flex; gap: 20px;">
                                <div style="flex: 1;">
                                    {donut_productos_activ_html}  <!-- Gráfico de donut: Productos Activos vs Inactivos -->
                                </div>
                                <div style="flex: 1;">
                                    {donut_productos_prodprov_html}  <!-- Gráfico de donut: Asignación de Proveedor a Producto -->
                                </div>
                                <div style="flex: 1;">
                                    {fig_sectores_productos_html}  <!-- Gráfico de sectores: Distribución de Familia Máster de Productos -->
                                </div>
                            </div>
                        </div>
                    """

                    # Segunda tarjeta con otro gráfico (puedes ajustar el contenido según tus necesidades)

                    tarjeta_inferior_1 = f"""
                        <div style="background-color: {fondo_rojo_claro}; padding: 20px; border-radius: 10px; display: flex; flex-direction: column; gap: 20px;">
                        <h3 style="font-weight: bold; text-align: center; color: black;">ALBARANES Y FACTURAS</h3>
                        <div style="display: flex; gap: 20px;">
                            <!-- Columna 1 -->
                            <div style="flex: 1; display: flex; flex-direction: column; justify-content: space-between;">
                                {gbarras_albaranes_html}
                                <div style="display: flex; gap: 10px;">
                                    <div style="flex: 1; padding: 10px; background-color: #0000FF; border-radius: 5px; height: 100px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                        <h5 style="font-weight: bold; text-align: center; color: white;">ALBARANES POR FOTO</h5>
                                        <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{total_albaranes_por_foto}</p>
                                    </div>
                                    <div style="flex: 1; padding: 10px; background-color: #000000; border-radius: 5px; height: 100px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                        <h5 style="font-weight: bold; text-align: center; color: white;">TOTAL ALBARANES</h5>
                                        <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{total_albaranes}</p>
                                    </div>
                                </div>
                            </div>
                            <!-- Columna 2 -->
                            <div style="flex: 1; display: flex; flex-direction: column; justify-content: space-between;">
                                {gbarras_albaranes_por_cerrar_html}
                                <div style="padding: 10px; background-color: #FF0000; border-radius: 5px; height: 100px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                    <h5 style="font-weight: bold; text-align: center; color: white;">ALBARANES POR CERRAR</h5>
                                    <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{total_albaranes_por_cerrar}</p>
                                </div>
                            </div>
                            <!-- Columna 3 -->
                            <div style="flex: 1; display: flex; flex-direction: column; justify-content: space-between;">
                                {gbarras_facturas_html}
                                <div style="display: flex; gap: 10px;">
                                    <div style="flex: 1; padding: 10px; background-color: #808080; border-radius: 5px; height: 100px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                        <h5 style="font-weight: bold; text-align: center; color: white;">TOTAL FACTURAS</h5>
                                        <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{total_facturas}</p>
                                    </div>
                                    <div style="flex: 1; padding: 10px; background-color: #808080; border-radius: 5px; height: 100px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                        <h5 style="font-weight: bold; text-align: center; color: white;">FACTURAS SIN LOCAL ASIGNADO</h5>
                                        <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{total_facturas_sin_local}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    """


                    # Tercera tarjeta con otro gráfico (puedes ajustar el contenido según tus necesidades)
                    tarjeta_inferior_2 = f"""
                        <div style="background-color: {fondo_rojo_claro}; padding: 20px; border-radius: 10px;">
                        <h3 style="font-weight: bold; text-align: center; color: black;">PEDIDOS Y PROVEEDORES</h3>

                        <div style="display: flex; gap: 20px;">
                            <!-- Columna 1 -->
                            <div style="flex: 5.5; display: flex; flex-direction: column; justify-content: space-between;">
                                <div style="display: flex; flex-direction: row; align-items: center; justify-content: center; gap: 20px;">
                                    <div style="flex: 1;">
                                        <p style="text-align: center;"> TOP 10 PROVEEDORES </p>
                                        {tabla_html_top}
                                    </div>
                                    <div style="flex: 1;">
                                        {donut_proveedores_activ_html}
                                    </div>
                                </div>
                                <div style="display: flex; gap: 10px; margin-top: 20px;">
                                    <div style="flex: 1; padding: 10px; background-color: #0000FF; border-radius: 5px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                        <h5 style="font-weight: bold; text-align: center; color: white;">TOTAL PROVEEDORES</h5>
                                        <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{total_proveedores}</p>
                                    </div>
                                    <div style="flex: 1; padding: 10px; background-color: #000000; border-radius: 5px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                                        <h5 style="font-weight: bold; text-align: center; color: white;">PROVEEDORES ACTIVOS</h5>
                                        <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{total_proveedores_activos}</p>
                                    </div>
                                </div>
                            </div>
                            <!-- Columna 2 -->
                            <div style="flex: 4.5; display: flex; flex-direction: column; justify-content: space-between;">
                                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 20px;">
                                    <div style="flex: 1;">
                                        {tabla_estado_pedidos_html}
                                    </div>
                                    <div style="flex: 1;">
                                        {fig_sectores_estado_pedidos_html}
                                    </div>
                                </div>
                                <div style="padding: 10px; background-color: #FF0000; border-radius: 5px; display: flex; align-items: center; justify-content: center; flex-direction: column; margin-top: 20px;">
                                    <h5 style="font-weight: bold; text-align: center; color: white;">TOTAL PEDIDOS</h5>
                                    <p style="font-size: 26px; font-weight: bold; text-align: center; margin: 0; color: white;">{total_pedidos}</p>
                                </div>
                            </div> 
                        </div>
                        </div>
                    """
                    

                    # Añadir las tarjetas al panel de acordeón
                    items.append(
                        ui.accordion_panel(
                            f"{modulo}".upper(),
                            ui.div(
                                ui.p(content),
                                ui.div(
                                    ui.HTML(tarjeta_superior),
                                    style="width: 100%; margin-bottom: 20px;"  # Ocupa todo el ancho y añade espacio debajo
                                ),
                                ui.div(
                                    ui.HTML(tarjeta_inferior_1),
                                    style="width: 100%; margin-bottom: 20px;"  # Ocupa todo el ancho y añade espacio debajo
                                ),
                                ui.div(
                                    ui.input_select(
                                        id="opcion_selector",  # Identificador único para el input
                                        label="Criterio de Ordenación",  # Etiqueta del selector
                                        choices={"N": "Número de pedidos", "B": "Importe promedio medio", "C": "Suma Importe Pedidos", "D": "Tiempo Medio Entrega"},  # Opciones disponibles
                                        selected="A"  # Opción preseleccionada
                                    ),
                                    ui.HTML(tarjeta_inferior_2),
                                    style="width: 100%;"  # Ocupa todo el ancho
                                )
                            ),
                            class_="accordion-panel-enabled"
                        )
                    )
                    end_compras = time.time()
                    time_compras = end_compras - start_compras
                    print("T-COMPRAS:", time_compras)
                    





# ----------------------------------------------------------------------------------------------------------



                elif modulo == 'cocina':
                    scatter_html = generate_scatter_plot().to_html(full_html=False)
                    items.append(
                        ui.accordion_panel(
                            f"{modulo}".upper(),
                            ui.div(
                                ui.p(content),
                                ui.HTML(scatter_html)
                            ),
                            class_="accordion-panel-enabled"
                        )
                    )

                else:
                    chart_html = generate_example_chart().to_html(full_html=False)
                    table_html = generate_example_table().to_html(full_html=False)
                    items.append(
                        ui.accordion_panel(
                            f"{modulo}".upper(),
                            ui.div(
                                ui.p(content),
                                ui.div(
                                    ui.HTML(chart_html),
                                    ui.HTML(table_html),
                                    style="display: flex; justify-content: space-around;"
                                )
                            ),
                            class_="accordion-panel-enabled"
                        )
                    )



            elif cliente_data[modulo].iloc[0] == 0:
                estado = "no contratado"
                background_color = "#FECACA"  # Rojo claro
                content = "Módulo no contratado"
                if modulo == 'kds' or modulo == 'amparo':
                    # URL directo a la imagen
                    image_url = "https://yurest-prod.s3.eu-west-3.amazonaws.com/perfil_cliente/PjlNWpwa2IHQv0PFnb2VgVTTqTdhpIao5PmgxGFj.svg"
                    historia = f"Esta es la historia de {modulo.upper()}: YUREEEEEEEST"
                    items.append(
                        ui.accordion_panel(
                            f"{modulo}".upper() + " (NO CONTRATADO)",
                            ui.div(
                                ui.p(content),
                                ui.div(
                                    ui.HTML(f"<img src='{image_url}' style='width:100%; max-width:400px;'>"),
                                    ui.p(historia),
                                    style=f"color: red; background-color: {background_color}; padding: 10px; border-radius: 5px;"  # Estilo de fondo y color de texto para módulo no contratado
                                )
                            ),
                            class_="accordion-panel-disabled"
                        )
                    )
                else:
                    items.append(
                        ui.accordion_panel(
                            f"{modulo.upper()} (NO CONTRATADO)",
                            ui.div(
                                ui.p(content),
                                style="color: red; background-color: #FECACA; padding: 10px; border-radius: 5px;"  # Estilo de fondo y color de texto para módulo no contratado
                            ),
                            class_="accordion-panel-disabled"
                        )
                    )
        else:
            title_red = f"<span style='color: red;'>{modulo.upper()} (NO CONTRATADO)</span>"
            items.append(
                ui.accordion_panel(
                    ui.HTML(title_red),  # Pasar el título como HTML
                    ui.div(
                        ui.p("Módulo no contratado", style="color: red;")
                    ),
                    class_="accordion-panel-disabled"
                )
            )

    return items






# Definir la interfaz de usuario
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style("""
            .accordion-panel-enabled .accordion-button {
                background-color: #90EE90 !important;  /* Verde claro */
                color: black !important;
                text-transform: uppercase;  /* Convertir texto a mayúsculas */
            }
            .accordion-panel-enabled .accordion-button:hover {
                background-color: #77DD77 !important;  /* Verde claro más oscuro */
            }
            .accordion-panel-disabled .accordion-button {
                background-color: #FFCCCB !important;  /* Rojo claro */
                color: black !important;
                cursor: not-allowed !important;
                pointer-events: none !important;
                text-transform: uppercase;  /* Convertir texto a mayúsculas */
            }
            .accordion-panel-disabled .accordion-button:hover {
                background-color: #FFA07A !important;  /* Rojo claro más oscuro */
            }
            .panel-content {
                padding: 0 18px;
                background-color: white;
                display: none;
                overflow: hidden;
            }
            .sidebar-fixed {
                position: fixed;
                top: 0;
                left: 0;
                width: 20%;
                height: 100%;
                overflow-y: auto;
                background-color: #f8f9fa;
                padding: 20px;
                border-right: 1px solid #dee2e6;
            }
            .main-content {
                margin-left: 20%;
                padding: 20px;
            }
            .spacer {
                margin-bottom: 20px;
            }
        """)
    ),
    ui.div(
        ui.div(
            ui.page_sidebar(
                ui.sidebar(
                    ui.div(
                        ui.input_select(
                            "cliente_selector", "Selecciona un Cliente",
                            choices={row['id']: row['nombre_comercial'] for index, row in df_clientes.iterrows()},
                            selected=1 
                        ),
                        ui.input_date_range("rango_fechas_seguimiento", "Selecciona un Rango de Fechas:",
                                            start=fecha_inicio_rango, end=fecha_actual),
                        class_="spacer"
                    ),
                    ui.div(
                        ui.output_ui("locales_list"),
                        class_="spacer"
                    ),
                    ui.div(
                        ui.input_checkbox_group(
                            "opciones_seleccionadas",  # ID del input
                            "Selecciona múltiples opciones:",
                            choices=['cocina', 'rrhh', 'almacenes', 'tareas', 'dashboards', 'compras', 'finanzas', 'ocr','appcc','comunicacion','formacion', 'kds', 'basicos', 'amparo', 'documentos']           # Lista de opciones a mostrar como checkboxes
                        )
                    ),
                    ui.div(
                        ui.markdown("Esto es una interfaz para revisar la usabilidad de los módulos de Yurest Solutions SL. Aquí puedes gestionar los módulos contratados por los clientes y visualizar información detallada sobre cada uno de ellos."),
                        class_="spacer"
                    )
                )
            ),
            class_="sidebar-fixed"
        ),
        ui.div(
            ui.div(
                ui.markdown("#### Información del Cliente:"),
                ui.output_ui("accordion_ui"),
                ui.output_text_verbatim("acc_cliente_val", placeholder=True)
            ),
            class_="main-content"
        )
    )
)

# modulos_presentes = ['cocina',  'rrhh', 'almacenes', 'tar', 'dashboards', 'compras', 'finanzas', 'ocr','formacion',  'kds', 'basicos', 'amparo', 'documentos']

# Definir el servidor
def server(input: Inputs, output: Outputs, session: Session):

    @output
    @render.ui
    def accordion_ui():
        cliente_id = int(input.cliente_selector())  # Asegurarse de que cliente_id es un entero
        fecha_inicio_cli = input.rango_fechas_seguimiento()[0]
        fecha_fin_cli = input.rango_fechas_seguimiento()[1]
        print(f"Cliente ID seleccionado: {cliente_id}")  # Mensaje de depuración
        cliente_data = df_clientes[df_clientes['id'] == cliente_id]
        print(f"Datos del cliente seleccionado:\n{cliente_data}")
        print("FECHASSSSS", fecha_inicio_cli, fecha_fin_cli )
        lista_modulos = list(input.opciones_seleccionadas())
        print(lista_modulos)  # Mensaje de depuración
        accordion_items = make_items(cliente_data, fecha_inicio_cli, fecha_fin_cli, lista_modulos)
        return ui.accordion(*accordion_items, id="acc_cliente", multiple=True, active=None)

    @output
    @render.ui
    def locales_list():
        cliente_id = int(input.cliente_selector())
        datos_compras = obtener_datos_compras(cliente_id)
        if not datos_compras.empty:
            locales_html = "<b>Locales:</b><br>" + "<br>".join(datos_compras['nombre_local'])
            return ui.HTML(locales_html)
        return ui.HTML("<b>Locales:</b><br>No hay locales disponibles")

    @render.text
    def acc_cliente_val():
        return "Cliente seleccionado: " + str(input.cliente_selector())

# Crear la aplicación
app = App(app_ui, server)


if __name__ == "__main__":
    app.run()

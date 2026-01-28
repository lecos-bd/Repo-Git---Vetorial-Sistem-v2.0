import numpy as np
import plotly.graph_objects as go
import data as dt
import numpy as np

# Carregamento do DataFrame a partir do módulo data
df = dt.df

def esferica_para_cartesiana(r, theta, phi):
    """
    Converte coordenadas esféricas para cartesianas.
    Garante o retorno de arrays para compatibilidade com o Plotly.
    """
    t_rad = np.radians(theta)
    p_rad = np.radians(phi)
    x = r * np.sin(t_rad) * np.cos(p_rad)
    y = r * np.sin(t_rad) * np.sin(p_rad)
    z = r * np.cos(t_rad)
    return [np.atleast_1d(a) for a in np.broadcast_arrays(x, y, z)]

def gerar_grafico(estado1, ano1, estado2=None, ano2=None):

    def obter_coord(estado, ano):
        # Conversão robusta para string para evitar erros de tipo
        dados = df.loc[(df['Estado'].astype(str) == str(estado)) & (df['Ano'].astype(str) == str(ano))]
        if dados.empty:
            return [0, 0, 0]
        eq = dados['Equidade'].values[0]
        am = dados['Ambiental'].values[0]
        se = dados['Segurança'].values[0]
        return [eq, se, am]

    def obter_raio(vetor):
        if vetor is None:
            return None
        return np.linalg.norm(vetor)

    # ---- Métricas e ângulos ----
    def angulo(coord1, coord2):
        coord1, coord2 = np.array(coord1), np.array(coord2)
        norm_v1 = np.linalg.norm(coord1)
        norm_v2 = np.linalg.norm(coord2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0
        return np.degrees(np.arccos(np.clip(
            np.dot(coord1, coord2) / (norm_v1 * norm_v2),
            -1, 1
        )))

    # --- TETRAEDRO (Formato Antigo: Sólido) ---
    def gerar_tetraedro(nome, cor, vetor, opacidade=0.6):
        base = np.array([[0,0,0], [vetor[0],0,0], [0,vetor[1],0], [0,0,vetor[2]]])
        faces = [[0,1,2], [0,1,3], [0,2,3], [1,2,3]]
        x, y, z = base[:,0], base[:,1], base[:,2]

        mesh = go.Mesh3d(
            x=x, y=y, z=z,
            i=[f[0] for f in faces],
            j=[f[1] for f in faces],
            k=[f[2] for f in faces],
            color=cor,
            opacity=opacidade,
            name=nome,
            hoverinfo='skip',
            showscale=False,
            showlegend=False
        )
        pontos = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=4, color=cor),
            name=f"Tetrahedron Vertices of {nome}",
            legendgroup=nome, 
            showlegend=False,
            hovertemplate=(
                f"{nome}<br>" +
                "x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f}<extra></extra>"
            )
        )
        return [mesh, pontos]
    
    # --- ARCOS (Formato Antigo: Setor Preenchido) ---
    def adicionar_arco_angulo(nome, coord1, coord2, cor_arco, cor_angulo, position, escala, opacidade):
        v1_norm = np.array(coord1) / np.linalg.norm(coord1) if np.linalg.norm(coord1) > 0 else np.array([0,0,0])
        v2_norm = np.array(coord2) / np.linalg.norm(coord2) if np.linalg.norm(coord2) > 0 else np.array([0,0,0])
        
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
        omega = np.arccos(dot_product)
        valor_angulo_graus = np.degrees(omega)

        if omega < 1e-6 or np.linalg.norm(coord1) == 0 or np.linalg.norm(coord2) == 0:
            return []

        N_pontos = 20 
        vertices = [[0, 0, 0]] 
        sin_omega = np.sin(omega)
        
        for i in range(N_pontos + 1):
            t = i / N_pontos
            a = np.sin((1 - t) * omega) / sin_omega
            b = np.sin(t * omega) / sin_omega
            interp_vec = (a * v1_norm + b * v2_norm)
            vertices.append(interp_vec * escala)
        
        vertices = np.array(vertices)
        
        i_idx, j_idx, k_idx = [], [], []
        for idx in range(1, N_pontos + 1):
            i_idx.append(0)
            j_idx.append(idx)
            k_idx.append(idx + 1)
            
        mesh_trace = go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=i_idx, j=j_idx, k=k_idx,
            color=cor_arco,
            opacity=opacidade,
            name=f"Ângulo {nome}",
            hoverinfo='skip',
            showlegend=False
        )

        mid_vec = v1_norm + v2_norm
        mid_norm = mid_vec / np.linalg.norm(mid_vec) if np.linalg.norm(mid_vec) > 0 else np.array([0,0,0])
        pos_texto = mid_norm * (escala * 1.1)

        if np.all(pos_texto == 0):
             return [mesh_trace]

        text_trace = go.Scatter3d(
            x=[pos_texto[0]], 
            y=[pos_texto[1]], 
            z=[pos_texto[2]],
            mode='text',
            text=[f"{valor_angulo_graus:.2f}°"],
            textfont=dict(color=cor_angulo, size=11, family="Aptos Black, sans-serif", weight="bold"),
            textposition=position,
            hoverinfo='skip',
            showlegend=False
        )

        return [mesh_trace, text_trace]

    # --- VETOR ORIGEM (Formato Antigo: Linha + Texto) ---
    def adicionar_vetor_origem(nome, cor, coord1, position):
        texto_label = f"{nome}"
        
        trace = go.Scatter3d(
            x=[0, coord1[0]], y=[0, coord1[1]], z=[0, coord1[2]],
            mode='lines+markers+text', 
            text=['', texto_label],     
            textposition=position, 
            textfont=dict(             
                size=14,
                color=cor,
                family="Aptos Black, sans-serif",
                weight="bold"
            ),
            line=dict(width=6, color=cor),
            marker=dict(size=6, color=cor),
            name=f"Vector {nome}",
            legendgroup=nome,
            showlegend=True,
            hovertemplate=(
                f"{nome}<br>" +
                "x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f}<extra></extra>"
            )
        )
        return [trace]

    # --- GRID ESFÉRICO (EXTERNO) ---
    def desenhar_grade_primeiro_octante(raio_max):
        grade_traces = []
        # Meridianos (0 a 90 graus apenas)
        for p in range(0, 91, 15):
            t_vals = np.linspace(0, 90, 50)
            x, y, z = esferica_para_cartesiana(raio_max, t_vals, p)
            grade_traces.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', 
                                           line=dict(color='lightgrey', width=2), 
                                           showlegend=False, hoverinfo='skip'))
        # Paralelos (0 a 90 graus apenas)
        for t in range(0, 91, 15):
            p_vals = np.linspace(0, 90, 50)
            x, y, z = esferica_para_cartesiana(raio_max, t, p_vals)
            grade_traces.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', 
                                           line=dict(color='lightgrey', width=2), 
                                           showlegend=False, hoverinfo='skip'))
        return grade_traces

    # --- GRIDS PLANOS INTERNOS (Preenche as "paredes" do quadrante) ---
    def desenhar_grades_planas(raio_max):
        traces = []
        grid_color = 'grey' # Bem sutil
        grid_width = 0.75

        # --- PLANO XY (Z=0) ---
        # Arcos concêntricos
        for r in np.linspace(0, raio_max, 6):
            p_vals = np.linspace(0, 90, 50)
            x = r * np.cos(np.radians(p_vals))
            y = r * np.sin(np.radians(p_vals))
            z = np.zeros_like(x)
            traces.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=grid_color, width=grid_width), opacity=0.25, showlegend=False, hoverinfo='skip'))
        # Raios
        for p in range(0, 91, 15):
            rad = np.radians(p)
            x = [0, raio_max * np.cos(rad)]
            y = [0, raio_max * np.sin(rad)]
            z = [0, 0]
            traces.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=grid_color, width=grid_width), opacity=0.25, showlegend=False, hoverinfo='skip'))

        # --- PLANO XZ (Y=0) ---
        # Arcos concêntricos
        for r in np.linspace(0, raio_max, 6):
            t_vals = np.linspace(0, 90, 50)
            x = r * np.sin(np.radians(t_vals))
            y = np.zeros_like(x)
            z = r * np.cos(np.radians(t_vals))
            traces.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=grid_color, width=grid_width), opacity=0.25, showlegend=False, hoverinfo='skip'))
        # Raios
        for t in range(0, 91, 15):
            rad = np.radians(t)
            x = [0, raio_max * np.sin(rad)]
            y = [0, 0]
            z = [0, raio_max * np.cos(rad)]
            traces.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=grid_color, width=grid_width), opacity=0.25, showlegend=False, hoverinfo='skip'))

        # --- PLANO YZ (X=0) ---
        # Arcos concêntricos
        for r in np.linspace(0, raio_max, 6):
            t_vals = np.linspace(0, 90, 50)
            x = np.zeros_like(t_vals)
            y = r * np.sin(np.radians(t_vals))
            z = r * np.cos(np.radians(t_vals))
            traces.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=grid_color, width=grid_width), opacity=0.25, showlegend=False, hoverinfo='skip'))
        # Raios
        for t in range(0, 91, 15):
            rad = np.radians(t)
            x = [0, 0]
            y = [0, raio_max * np.sin(rad)]
            z = [0, raio_max * np.cos(rad)]
            traces.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=grid_color, width=grid_width), opacity=0.25, showlegend=False, hoverinfo='skip'))

        return traces

    # --- DESENHAR EIXOS X, Y, Z COM DERMACAÇÕES (1 a 10) ---
    def desenhar_eixos(comprimento):
        eixos_traces = []
        tick_vals = list(range(1, 18))
        tick_len = 0.3  # Tamanho do tracinho para as marcações
        
        # --- EIXO X (Equidade) ---
        eixos_traces.append(go.Scatter3d(
            x=[0, comprimento], y=[0, 0], z=[0, 0],
            mode='lines+text', text=['', 'Energy Equity'], textposition='top left',
            textfont=dict(family="Aptos Black, sans-serif", size=12, color="black"),
            line=dict(color='gray', width=4), showlegend=False, hoverinfo='skip'
        ))
        # Ticks X (Desenhados apontando para Y negativo)
        x_tick_x, x_tick_y, x_tick_z = [], [], []
        for val in tick_vals:
            x_tick_x.extend([val, val, None])
            x_tick_y.extend([0, -tick_len, None])
            x_tick_z.extend([0, 0, None])
        eixos_traces.append(go.Scatter3d(x=x_tick_x, y=x_tick_y, z=x_tick_z, mode='lines', line=dict(color='black', width=0.25), showlegend=False, hoverinfo='skip'))
        # Números X
        eixos_traces.append(go.Scatter3d(
            x=tick_vals, y=[-tick_len*2.5]*17, z=[0]*17,
            mode='text', text=[str(v) for v in tick_vals],
            textfont=dict(size=8.5, color='gray', family="Aptos Black, sans-serif"),
            showlegend=False, hoverinfo='skip'
        ))
        
        # --- EIXO Y (Segurança) ---
        eixos_traces.append(go.Scatter3d(
            x=[0, 0], y=[0, comprimento], z=[0, 0],
            mode='lines+text', text=['', 'Energy Security'], textposition='top right',
            textfont=dict(family="Aptos Black, sans-serif", size=12, color="black"),
            line=dict(color='gray', width=4), showlegend=False, hoverinfo='skip'
        ))
        # Ticks Y (Desenhados apontando para X negativo)
        y_tick_x, y_tick_y, y_tick_z = [], [], []
        for val in tick_vals:
            y_tick_x.extend([0, -tick_len, None])
            y_tick_y.extend([val, val, None])
            y_tick_z.extend([0, 0, None])
        eixos_traces.append(go.Scatter3d(x=y_tick_x, y=y_tick_y, z=y_tick_z, mode='lines', line=dict(color='black', width=0.25), showlegend=False, hoverinfo='skip'))
        # Números Y
        eixos_traces.append(go.Scatter3d(
            x=[-tick_len*2.5]*17, y=tick_vals, z=[0]*17,
            mode='text', text=[str(v) for v in tick_vals],
            textfont=dict(size=8.5, color='grey', family="Aptos Black, sans-serif"),
            showlegend=False, hoverinfo='skip'
        ))
        
        # --- EIXO Z (Ambiental) ---
        eixos_traces.append(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, comprimento],
            mode='lines+text', text=['', 'Environment Sustainability'], textposition='top right',
            textfont=dict(family="Aptos Black, sans-serif", size=12, color="black"),
            line=dict(color='grey', width=4), showlegend=False, hoverinfo='skip'
        ))
        # Ticks Z (Desenhados apontando para X negativo)
        z_tick_x, z_tick_y, z_tick_z = [], [], []
        for val in tick_vals:
            z_tick_x.extend([0, -tick_len, None])
            z_tick_y.extend([0, 0, None])
            z_tick_z.extend([val, val, None])
        eixos_traces.append(go.Scatter3d(x=z_tick_x, y=z_tick_y, z=z_tick_z, mode='lines', line=dict(color='black', width=0.25), showlegend=False, hoverinfo='skip'))
        # Números Z
        eixos_traces.append(go.Scatter3d(
            x=[-tick_len*2.5]*17, y=[0]*17, z=tick_vals,
            mode='text', text=[str(v) for v in tick_vals],
            textfont=dict(size=8.5, color='grey', family="Aptos Black, sans-serif"),
            showlegend=False, hoverinfo='skip'
        ))

        return eixos_traces

    # Obtém as coordenadas
    coord_ideal = [10,10,10]
    raio_esfera = np.linalg.norm(coord_ideal) # Raio baseado na norma do ideal

    coord1 = obter_coord(estado1, ano1) if estado1 and ano1 else None
    coord2 = obter_coord(estado2, ano2) if estado2 and ano2 else None

    # Vetores Polares (para logica interna)
    vetor1 = [obter_raio(coord1), angulo(coord1,np.array([1,0,0])), angulo(coord1,np.array([0,0,1]))] if coord1 else None
    vetor2 = [obter_raio(coord2), angulo(coord2,np.array([1,0,0])), angulo(coord2,np.array([0,0,1]))] if coord2 else None

    fig = go.Figure()

    # 1. Grid Esférico (Externo)
    for gt in desenhar_grade_primeiro_octante(raio_esfera): 
        fig.add_trace(gt)
    
    # 2. Grades Planas Internas (NOVO - Preenche o interior)
    for gp in desenhar_grades_planas(raio_esfera):
        fig.add_trace(gp)
    
    # 3. Eixos (Comprimento exato do raio + Marcações)
    for ax in desenhar_eixos(raio_esfera):
        fig.add_trace(ax)

    # Ideal (Vetor)
    for tr in adicionar_vetor_origem("IDEAL", "gray", coord_ideal, position='top center'):
       fig.add_trace(tr)
    for tr in gerar_tetraedro("Ideal Tetrahedron", "black", coord_ideal, opacidade=0.08):
        fig.add_trace(tr)

    # State 1
    if vetor1:
        for tr in adicionar_vetor_origem(f"{estado1} {ano1}", "black", coord1, position='top right'):
            fig.add_trace(tr)
        for tr in gerar_tetraedro(f"{estado1} {ano1}", "black", coord1, opacidade=0.1):
            fig.add_trace(tr)

        # Arcos Eixos
        for tr in adicionar_arco_angulo(f"{estado1}-X", coord1, np.array([1, 0, 0]), "black", "black", position='top right', opacidade=0.35, escala=2.0):
            fig.add_trace(tr)
        for tr in adicionar_arco_angulo(f"{estado1}-Z", coord1, np.array([0, 0, 1]), "black", "black", position='top center', opacidade=0.35, escala=2.5):
            fig.add_trace(tr)

    # State 2
    if vetor2:
        for tr in adicionar_vetor_origem(f"{estado2} {ano2}", "gray", coord2, position='top left'):
            fig.add_trace(tr)

        # Arco Ideal (Corrigido para usar coord_ideal)
        for tr in adicionar_arco_angulo(f"{estado2}-Ideal", coord2, coord_ideal, "lightgray", "black", position='top right', opacidade=1, escala=4.0):
            fig.add_trace(tr)

    # Layout Final
    titulo_texto = f"Vetorial System - {estado1} ({ano1})"
    if vetor1 and vetor2:
        titulo_texto = f"{estado1} ({ano1}) vs {estado2} ({ano2})"
    elif vetor1:
        titulo_texto = f"VECTOR {estado1} {ano1} (r = {vetor1[0]:.2f} , θ = {vetor1[1]:.2f}° , φ = {vetor1[2]:.2f}°)"

    margem_range = raio_esfera * 1.15
    range_eixos = [-1, margem_range] # Pequeno negativo para ver a origem

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=range_eixos),
            yaxis=dict(visible=False, range=range_eixos),
            zaxis=dict(visible=False, range=range_eixos),
            bgcolor="white", 
            aspectmode='cube'
        ),
        scene_camera=dict(eye=dict(x=0.5, y=-2, z=0)),
        title=dict( 
            text=titulo_texto,
            font=dict(family="Aptos Black, sans-serif", size=18, color="black", weight="bold"),
            x=0.5, xanchor='center'
        ),
        width=None, height=700, autosize=True,
        showlegend=True,
        legend=dict(x=0.85, y=0.85, xanchor="center", yanchor="middle", font=dict(size=8), bgcolor='rgba(255,255,255,0.5)'),
        margin=dict(l=90, r=0, t=80, b=30)
    )

    def formatar_vetor(v): return f"(x={v[0]:.2f}, y={v[1]:.2f}, z={v[2]:.2f})"
    eixo_x, eixo_y, eixo_z = [1,0,0], [0,1,0], [0,0,1]
    
    resultado = {"vetor_ideal": f"Ideal: {formatar_vetor(coord_ideal)}"}

    if vetor1:
        resultado.update({
            "vetor_real_1": f"{estado1} {ano1}: {formatar_vetor(coord1)}",
            "angulo_ideal_1": f"Ângulo entre {estado1} e Ideal: {angulo(coord1, coord_ideal):.2f}°",
            "angulo_x_1": f"Ângulo entre {estado1} e eixo X: {angulo(coord1, eixo_x):.2f}°",
            "angulo_y_1": f"Ângulo entre {estado1} e eixo Y: {angulo(coord1, eixo_y):.2f}°",
            "angulo_z_1": f"Ângulo entre {estado1} e eixo Z: {angulo(coord1, eixo_z):.2f}°",
        })

    if vetor2:
        resultado.update({
            "vetor_real_2": f"{estado2} {ano2}: {formatar_vetor(coord2)}",
            "angulo_ideal_2": f"Ângulo entre {estado2} e Ideal: {angulo(coord2, coord_ideal):.2f}°",
            "angulo_x_2": f"Ângulo entre {estado2} e eixo X: {angulo(coord2, eixo_x):.2f}°",
            "angulo_y_2": f"Ângulo entre {estado2} e eixo Y: {angulo(coord2, eixo_y):.2f}°",
            "angulo_z_2": f"Ângulo entre {estado2} e eixo Z: {angulo(coord2, eixo_z):.2f}°",
        })

    return fig.to_html(full_html=False, include_plotlyjs='cdn'), resultado


def gerar_grafico_radar(estado1, ano1, estado2=None, ano2=None):
    """
    Gera um gráfico de radar comparativo com nomes de eixos personalizados e rotação ajustada.
    """
    def obter_vetor_dict(estado, ano):
        dados = df.loc[(df['Estado'].astype(str) == str(estado)) & (df['Ano'].astype(str) == str(ano))]
        if dados.empty: 
            return {"Energy Equity": 0, "Energy Security": 0, "Environmental Sustainability": 0}
        return {
            "Energy Equity": dados['Equidade'].values[0],
            "Energy Security": dados['Segurança'].values[0],
            "Environmental Sustainability": dados['Ambiental'].values[0]
        }

    v1 = obter_vetor_dict(estado1, ano1) if (estado1 and ano1) else None
    v2 = obter_vetor_dict(estado2, ano2) if (estado2 and ano2) else None

    # Adicionando <br> para "empurrar" o texto para longe do gráfico
    # Equity (Top): <br> depois do texto
    # Security/Sustainability (Bottom): <br> antes do texto
    categorias_br = [
        "<br><br>Energy Equity", 
        "Energy Security<br><br>", 
        "<br><br>Environmental Sustainability<br><br>"
    ]
    
    # Mapeamento interno para facilitar a busca de valores
    cat_map = {
        "<br><br>Energy Equity": "Energy Equity",
        "Energy Security<br><br>": "Energy Security",
        "<br><br>Environmental Sustainability<br><br>": "Environmental Sustainability"
    }
    
    # Nomes dos eixos modificados para inglês conforme padrão do sistema
    categorias = ["Energy Equity", "Energy Security", "Environmental Sustainability"]
    theta = categorias + [categorias[0]] 
    
    fig = go.Figure()

    # 1. TRACE IDEAL
    r_ideal = [10, 10, 10, 10]
    fig.add_trace(go.Scatterpolar(
        r=r_ideal,
        theta=theta,
        fill='none',
        name="Ideal (10,10,10)",
        line=dict(color='black', dash='dash', width=2)
    ))

    # 2. ESTADO 1
    if v1 and any(v1.values()):
        r1 = [v1[c] for c in categorias]
        r1.append(r1[0])
        fig.add_trace(go.Scatterpolar(
            r=r1,
            theta=theta,
            fill='toself',
            name=f"{estado1} {ano1}",
            line=dict(color='black'),
            opacity=0.7
        ))  

    # 3. ESTADO 2
    if v2 and any(v2.values()):
        r2 = [v2[c] for c in categorias]
        r2.append(r2[0])
        fig.add_trace(go.Scatterpolar(
            r=r2,
            theta=theta,
            fill='toself',
            name=f"{estado2} {ano2}",
            line=dict(color='darkgray'),
            opacity=0.8
        ))

    fig.update_layout(
        polar=dict(
            domain=dict(x=[0.1, 0.9], y=[0.1, 0.9]),
            radialaxis=dict(visible=True, range=[0, 10], gridcolor="darkgray"),
            angularaxis=dict(
                gridcolor="darkgray", 
                tickfont=dict(family="Aptos Black, sans-serif", size=14, color="black", weight="bold"),
                rotation=-30, # Ajusta a rotação para que o primeiro item fique no topo
                direction="clockwise"
            )
        ),
        showlegend=True,
        title=dict(
            text="Trilemma Comparison (2D Radar)",
            font=dict(family="Aptos Black, sans-serif", size=20, color="black", weight="bold"),
            x=0.5, xanchor='center'
        ),
        width=None, height=700, autosize=True,
        margin=dict(l=80, r=80, t=100, b=80),
        legend=dict(
            x=1.15, # Posiciona a legenda mais à direita, próxima à borda
            y=1, 
            xanchor="right", 
            yanchor="top",
            font=dict(size=10, family="Aptos Black, sans-serif"), 
            bgcolor='rgba(255,255,255,0.5)', 
            bordercolor='black', 
            borderwidth=1
        )
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')
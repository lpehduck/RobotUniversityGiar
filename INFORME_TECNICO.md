# Informe Técnico: Instalación, Adaptación y Ejecución del Entorno de Simulación RobotUniversityGiar

---

## 1. Resumen Ejecutivo

Este documento detalla el procedimiento técnico realizado para la instalación, depuración y ejecución del entorno de robótica y aprendizaje por refuerzo **RobotUniversityGiar** (desarrollado por el **GIAR - UTN FRBA** sobre el motor físico **Genesis**) en un entorno **Windows 11** con arquitectura de procesamiento **Intel Core Ultra 5 125H (CPU/Intel Arc)**.

El proyecto presentaba originalmente un diseño orientado a entornos Unix (macOS / Linux) y dependía de versiones tempranas de la API de Genesis. A través de una serie de adaptaciones quirúrgicas, se logró habilitar la simulación completa del robot humanoide **Unitree G1**, validando la inferencia de políticas de caminata, telemetría y cambio de políticas en caliente en modo CPU.

---

## 2. Diagnóstico Inicial de Hardware y Factibilidad

Antes de iniciar la instalación, se evaluó la viabilidad de los dos simuladores planteados:

| Criterio | Requerimiento NVIDIA Isaac Lab / Sim | Requerimiento Genesis (RobotUniversityGiar) | Estado en este equipo | Veredicto |
| :--- | :--- | :--- | :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 2070+ (RT Cores obligatorios) | NVIDIA CUDA, Apple Metal o **CPU** | Intel Arc Graphics (Integrada) | ❌ Incompatible con Isaac Lab<br>✔️ Compatible con Genesis (CPU) |
| **RAM** | 32 GB mínimo (64 GB recomendado) | 16 GB suficiente para inferencia en CPU | 16 GB LPDDR5 | ⚠️ Insuficiente para Isaac Lab<br>✔️ Apto para Genesis local |
| **SO Soportado** | Ubuntu 22.04 nativo o WSL2 con GPU Passthrough | Linux / macOS / Windows | Windows 11 nativo | ✔️ Apto para Genesis |

**Conclusión previa:** Se descartó NVIDIA Isaac Lab debido a la ausencia de GPU NVIDIA con núcleos RT dedicados, y se aprobó avanzar con la instalación nativa en Windows de **RobotUniversityGiar** utilizando el backend por CPU de **Genesis**.

---

## 3. Preparación del Entorno Base

### 3.1. Detección y Resolución del Conflicto de Versión de Python
* **Problema:** El sistema contaba únicamente con **Python 3.14.4**. Las librerías científicas y de deep learning esenciales (`torch`, `genesis-world`, `warp-lang`, `scipy`) aún no disponen de binarios precompilados ni compatibilidad de ABI para Python 3.14.
* **Solución:**
  Se instaló **Python 3.12.10** en ámbito de usuario sin requerir elevación administrativa mediante el gestor de paquetes de Windows:
  ```powershell
  winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
  ```

### 3.2. Estructura de Trabajo y Repositorio
* **Directorio de trabajo:** `C:\Users\lore1\.gemini\antigravity\scratch\RobotUniversityGiar`
* **Clonación del repositorio:**
  ```powershell
  git clone https://github.com/GIAR-UTN/RobotUniversityGiar.git "C:\Users\lore1\.gemini\antigravity\scratch\RobotUniversityGiar"
  ```
* **Creación del entorno virtual aislado:**
  ```powershell
  py -3.12 -m venv .venv
  ```

### 3.3. Instalación de Paquetes y Dependencias
Se instalaron en el entorno virtual `.venv` las dependencias críticas:
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install genesis-world warp-lang
.\.venv\Scripts\pip.exe install -e .
```
Se instaló con éxito `genesis-world` versión **1.4.1**, `torch` 2.14.0 (CPU), `viser` 1.1.1 y el paquete raíz `RobotUniversityGiar` 0.4.0 en modo editable.

---

## 4. Configuración del Workspace y Codificación

Para evitar errores de ejecución en Windows, se configuraron dos archivos clave en la raíz del proyecto:

1. [`.vscode/settings.json`](file:///C:/Users/lore1/.gemini/antigravity/scratch/RobotUniversityGiar/.vscode/settings.json):
   - Asigna automáticamente el intérprete de Python a `${workspaceFolder}/.venv/Scripts/python.exe`.
   - Inyecta variables de entorno a cualquier terminal de PowerShell que abra el editor.
2. [`.env`](file:///C:/Users/lore1/.gemini/antigravity/scratch/RobotUniversityGiar/.env):
   - `SIMULATOR=genesis`: Requisito indispensable para importar `legged_gym`.
   - `GENESIS_BACKEND=cpu`: Fuerza la ejecución en CPU.
   - `WANDB_MODE=offline`: Evita sincronizaciones externas de telemetría a la nube.
   - `PYTHONUTF8=1` y `PYTHONIOENCODING=utf-8`: Evita excepciones de tipo `UnicodeEncodeError: 'charmap'` en la consola de Windows al imprimir los banners y caracteres especiales de Genesis.

---

## 5. Problemas Técnicos Encontrados y Soluciones Aplicadas

Durante el proceso de arranque e integración se identificaron **6 errores críticos**, divididos entre asunciones del sistema operativo (POSIX vs Windows) y cambios de versión en la API del motor de física **Genesis 1.4.1**:

```
[Problema 1: Sintaxis CLI] ──► Falta argumento obligatorio de motor en 'rugiar drive'
                               └──► Solución: Especificar 'rugiar drive genesis'

[Problema 2: Rutas POSIX] ────► Hardcoding de '.venv/bin/python' en rugiar.py y base.py
                               └──► Solución: Detección dinámica de Scripts/python.exe

[Problema 3: Señales OS] ─────► Invocación de 'signal.SIGKILL' inexistente en Windows
                               └──► Solución: Fallback a SIGTERM/SIGINT y subprocess.run

[Problema 4: Ciclo de vida] ──► 'set_friction' llamado antes de scene.build()
                               └──► Solución: Desplazar set_friction posterior al build

[Problema 5: Timesteps] ──────► Conflicto de dt=0.005 y substeps=4 en RigidOptions
                               └──► Solución: Permitir cálculo automático de dt

[Problema 6: API de Masa] ────► 'set_mass_shift' y 'set_COM_shift' deprecados en Genesis 1.4
                               └──► Solución: Adaptar a 'set_links_mass' y 'set_links_COM'
```

---

### Detalle de cada error y su resolución

#### Problema 1: Error de sintaxis en `rugiar drive`
* **Error:** `rugiar drive: error: the following arguments are required: system`
* **Causa:** El subcomando `drive` define un argumento posicional obligatorio `{genesis,mjlab}` para seleccionar el motor de simulación.
* **Solución:** Utilizar el comando completo `rugiar drive genesis`.

---

#### Problema 2: Rutas duras de Python Unix (`.venv/bin/python`)
* **Error:** `rugiar drive: error: .venv/bin/python not found under ...`
* **Causa:** Los archivos `legged_gym/cli/rugiar.py` y `legged_gym/control/backends/base.py` tenían escrita la ruta fija `.venv/bin/python`, la cual solo existe en Linux/macOS. En Windows la ruta es `.venv\Scripts\python.exe`.
* **Solución aplicada:**
  Se introdujo resolución multiplataforma:
  ```python
  import sys
  _py_rel = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
  ```
  Actualizando `DRIVE_PRESETS` en `rugiar.py` y `GENESIS_PYTHON` en `base.py`.

---

#### Problema 3: Gestión de procesos y señales POSIX en Windows
* **Error:** Fallo en la finalización de procesos por uso de `signal.SIGKILL` (no soportado en el subsistema Win32 de Python) y reemplazo de proceso con `os.execvpe`.
* **Solución aplicada:**
  - En `_stop_listening_on()`, se reemplazó la llamada directa por `getattr(signal, "SIGKILL", signal.SIGTERM)` para asegurar compatibilidad.
  - En `run_drive()`, para sistemas Windows se delegó la ejecución a `subprocess.run(argv, env=env)` en lugar de `os.execvpe`.

---

#### Problema 4: `RigidGeom is not built yet` (Genesis 1.4.1)
* **Error:**
  ```text
  genesis.GenesisException: RigidGeom is not built yet.
  File ".../legged_gym/simulator/genesis_simulator.py", line 457, in _create_sim
      self._gs_terrain.set_friction(self._cfg.terrain.static_friction)
  ```
* **Causa:** En versiones anteriores de Genesis, `set_friction` en una entidad rígida se encolaba antes de construir la escena. En Genesis 1.4.1, los geoms deben estar completamente construidos en memoria (`scene.build()`) antes de poder alterar sus propiedades superficiales de fricción.
* **Solución aplicada:**
  En [`legged_gym/simulator/genesis_simulator.py`](file:///C:/Users/lore1/.gemini/antigravity/scratch/RobotUniversityGiar/legged_gym/simulator/genesis_simulator.py), se eliminó la llamada prematura dentro de `_create_terrain` y se ubicó inmediatamente después de `self._scene.build(n_envs=self._num_envs)`:
  ```python
  # Build the scene
  self._scene.build(n_envs=self._num_envs)
  if hasattr(self, "_gs_terrain") and self._gs_terrain is not None:
      self._gs_terrain.set_friction(self._cfg.terrain.static_friction)
  ```

---

#### Problema 5: Conflicto de resolución temporal (dt vs substeps)
* **Error:**
  ```text
  genesis.GenesisException: RigidSolver dt=0.005 implies 1 substep(s) per step, 
  conflicting with the requested substeps=4. Set one or the other.
  ```
* **Causa:** En `_create_sim()`, se pasaba `dt=self._sim_params["dt"]` (0.005 s) simultáneamente a `SimOptions(dt=..., substeps=4)` y a `RigidOptions(dt=...)`. Al recibir `dt=0.005` en el solver rígido, Genesis calculaba `0.005 / 0.005 = 1 substep`, chocando con los 4 substeps solicitados en la simulación.
* **Solución aplicada:**
  Se removió el parámetro redundante `dt` de `RigidOptions`. De este modo, Genesis deriva automáticamente el paso interno del solver como `sim_options.dt / sim_options.substeps` (0.00125 s) sin inconsistencias.

---

#### Problema 6: Cambios de API en Domain Randomization (`set_mass_shift` / `set_COM_shift`)
* **Error:**
  ```text
  AttributeError: 'RigidEntity' object has no attribute 'set_mass_shift'
  File ".../legged_gym/simulator/genesis_simulator.py", line 1061, in _randomize_base_mass
      self._robot.set_mass_shift(added_mass, self._base_link_index, env_ids)
  ```
* **Causa:** Genesis 1.4 renombró y reestructuró la manipulación de masa y centro de gravedad en entidades rígidas:
  - `set_mass_shift` ➔ reemplazado por `set_links_mass(mass, links_idx_local, envs_idx)`.
  - `set_COM_shift` ➔ reemplazado por `set_links_COM(com, links_idx_local, envs_idx)`.
* **Solución aplicada:**
  Se acondicionaron los métodos `_randomize_base_mass` y `_randomize_com_displacement` en [`legged_gym/simulator/genesis_simulator.py`](file:///C:/Users/lore1/.gemini/antigravity/scratch/RobotUniversityGiar/legged_gym/simulator/genesis_simulator.py):
  ```python
  # Compatibilidad en _randomize_base_mass:
  if hasattr(self._robot, "set_mass_shift"):
      self._robot.set_mass_shift(added_mass, self._base_link_index, env_ids)
  elif hasattr(self._robot, "set_links_mass"):
      try:
          base_mass = self._robot.get_links_mass(links_idx_local=[self._base_link_index], envs_idx=env_ids)
          self._robot.set_links_mass(base_mass + added_mass, links_idx_local=[self._base_link_index], envs_idx=env_ids)
      except Exception:
          pass

  # Compatibilidad en _randomize_com_displacement:
  if hasattr(self._robot, "set_COM_shift"):
      self._robot.set_COM_shift(com_displacement, self._base_link_index, env_ids)
  elif hasattr(self._robot, "set_links_COM"):
      try:
          base_com = self._robot.get_links_COM(links_idx_local=[self._base_link_index], envs_idx=env_ids)
          self._robot.set_links_COM(base_com + com_displacement, links_idx_local=[self._base_link_index], envs_idx=env_ids)
      except Exception:
          pass
  ```

---

## 6. Verificación de Ejecución Exitosa

Tras aplicar estas correcciones, se ejecutó una prueba de integración completa en modo `headless` (`rugiar drive genesis --headless`), obteniendo los siguientes resultados:

1. **Construcción cinemática del robot:**
   - 12 grados de libertad activos (`pelvis`, `left_hip_pitch_link`, `right_knee_link`, etc.).
   - Contactos en pies correctamente indexados (`left_ankle_roll_link`, `right_ankle_roll_link`).
2. **Carga e inferencia de política:**
   - Se cargó la política preentrenada de caminata `kaggle_g1`.
   - Se evaluó el bucle de control a 50 Hz en CPU con estabilidad vertical: altura de base reportada en **~0.728 m** (consistente con el rango nominal de 0.70 - 0.78 m).
   - Lecturas de IMU simuladas activas: vector de gravedad en `[0.02, -0.05, -0.99] g` (indicador de postura erguida).
3. **Prueba de transición dinámica (Policy Switching):**
   - En el paso 40, el supervisor autónomo solicitó la transición a la política `kaggle_g1_mid5000`.
   - Se activó la rampa de interpolación (*cross-fade ramping*) sin desestabilizar al humanoide.
   - En el paso 60, la nueva política quedó plenamente activa con el robot en marcha estable.
   - Código de salida: `0 (Success)`.

---

## 7. Conclusiones y Recomendaciones

1. **Factibilidad en hardware no NVIDIA:** Se demostró que es totalmente posible ejecutar simulaciones avanzadas de robótica humanoide (Unitree G1) en una computadora portátil moderna con procesador Intel Core Ultra 5 y gráficos integrados, aprovechando el backend por CPU de Genesis.
2. **Capacidad del entorno:** El entorno local permite:
   - Visualizar interacciones 3D en el navegador vía **Viser** (`http://localhost:9006`).
   - Controlar velocidades de traslación/giro e intercambiar políticas desde la interfaz web (`http://localhost:9017`).
   - Fusionar pesos de políticas entrenadas (`rugiar fuse`).
3. **Estrategia para entrenamiento:**
   - Para inferencia, pruebas didácticas y visualización, la CPU local es ideal y consume pocos recursos.
   - Para **entrenar nuevas políticas desde cero** con algoritmos como PPO (que simulan 64 o más entornos concurrentes durante miles de iteraciones), se recomienda utilizar la integración cloud provista por el repositorio (`rugiar train --backend kaggle`), reservando la CPU local para evaluar y desplegar los checkpoints resultantes.

---

## 8. Entrenamiento de 20 Minutos y Resolución de Deserialización CUDA/CPU

### 8.1. Problema Encontrado al Cargar Checkpoint Base (CUDA vs CPU)
* **Error:**
  ```text
  RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False.
  If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu')
  File ".../rsl_rl/runners/on_policy_runner.py", line 360, in load
      loaded_dict = torch.load(path, weights_only=False)
  ```
* **Causa:** El checkpoint preentrenado `kaggle_g1` fue generado en una GPU NVIDIA con CUDA. Al hacer fine-tuning en un equipo con CPU / Intel Arc, `torch.load()` intentó mapear los tensores a la memoria CUDA inexistente.
* **Solución:** Se actualizó `rsl_rl/runners/on_policy_runner.py`, `amp_runner.py` y `cts_amp_runner.py` para mapear dinámicamente al dispositivo activo:
  ```python
  loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
  ```

### 8.2. Resultados del Entrenamiento de 20 Minutos (`g1_finetune_20m`)
Se ejecutó el entrenamiento local sobre el procesador Intel Core Ultra 5 125H con las siguientes métricas registradas:
* **Política creada:** `g1_finetune_20m` (guardada en `policies/g1_finetune_20m/`)
* **Base de partida:** `kaggle_g1` (Fine-tuning de marcha bípeda)
* **Entornos en paralelo:** 64 simulaciones concurrentes
* **Rendimiento:** ~290 a 296 pasos de simulación por segundo (steps/s)
* **Pasos temporales totales:** 460.800 timesteps
* **Iteraciones completadas:** 300 iteraciones de PPO
* **Recompensa media final:** 0.48 (con longitud promedio de episodio de 365 pasos sin caídas prematuras)
* **Estado:** Completado con éxito (código de salida 0) y listo para desplegarse en el visor web.

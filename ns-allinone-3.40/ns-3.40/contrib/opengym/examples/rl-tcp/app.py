import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, url_for

app = Flask(__name__, template_folder='templates')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(BASE_DIR, 'test_model.py')
TRAINING_PLOTS_DIR = os.path.join(BASE_DIR, 'training_plots')
TEST_PLOTS_DIR = os.path.join(BASE_DIR, 'test_plots')
TRAIN_PLOT_PATH = os.path.join(TRAINING_PLOTS_DIR, 'plots.png')
TEST_PLOT_PATH = os.path.join(TEST_PLOTS_DIR, 'plots.png')
HISTORY_DIR = os.path.join(BASE_DIR, 'run_history')
HISTORY_FILE = os.path.join(HISTORY_DIR, 'runs.json')
PYTHON_EXECUTABLE = sys.executable

SUPPORTED_ARGS = [
    'train_episodes',
    'test_episodes',
    'duration',
    'step_time',
    'port',
    'seed',
    'load',
    'save',
    'buffer_capacity',
    'action_mode',
    'action0_scale',
    'action1_add',
    'action2_add',
    'action3_scale',
    'action4_scale',
    'action5_scale',
    'bottleneck_bandwidth',
    'bottleneck_delay',
    'access_bandwidth',
    'access_delay',
    'mtu',
    'nLeaf',
]

os.makedirs(TRAINING_PLOTS_DIR, exist_ok=True)
os.makedirs(TEST_PLOTS_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as fp:
        json.dump([], fp, indent=2)

NETANIM_BINARY = os.path.abspath(os.path.normpath(os.path.join(BASE_DIR, '..', '..', '..', '..','..', 'netanim-3.109', 'NetAnim')))
NETANIM_XML = os.path.abspath(os.path.normpath(os.path.join(BASE_DIR, '..', '..', '..', '..', 'TcpVariantsComparison-netanim.xml')))


def load_history():
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as fp:
            return json.load(fp)
    except (OSError, ValueError):
        return []


def save_history(entries):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as fp:
        json.dump(entries, fp, indent=2)


def record_history(entry):
    entries = load_history()
    entries.insert(0, entry)
    save_history(entries)
    return entries


def get_run_dir(run_id):
    return os.path.join(HISTORY_DIR, run_id)


def copy_plot(source, target):
    try:
        shutil.copy2(source, target)
    except OSError:
        pass


def build_plot_urls(run_id=None):
    if run_id:
        return {
            'training': url_for('get_history_plot', run_id=run_id, plot_type='training', _external=False),
            'test': url_for('get_history_plot', run_id=run_id, plot_type='test', _external=False),
        }
    return {
        'training': url_for('get_plot', plot_type='training', _external=False),
        'test': url_for('get_plot', plot_type='test', _external=False),
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run', methods=['POST'])
def run_model():
    form = request.form
    command = [PYTHON_EXECUTABLE, SCRIPT_PATH]
    args = {}

    for arg_name in SUPPORTED_ARGS:
        value = form.get(arg_name)
        if value is None:
            continue

        if arg_name in ['train_episodes', 'test_episodes', 'buffer_capacity', 'port', 'seed', 'mtu', 'nLeaf']:
            args[arg_name] = int(value)
            command.extend([f'--{arg_name}', str(args[arg_name])])
        elif arg_name in ['duration', 'step_time', 'action0_scale', 'action3_scale', 'action4_scale', 'action5_scale']:
            args[arg_name] = float(value)
            command.extend([f'--{arg_name}', str(args[arg_name])])
        elif arg_name in ['action1_add', 'action2_add']:
            args[arg_name] = int(value)
            command.extend([f'--{arg_name}', str(args[arg_name])])
        elif arg_name == 'action_mode':
            args[arg_name] = value.strip() or 'default'
            command.extend([f'--{arg_name}', args[arg_name]])
        elif arg_name in ['bottleneck_bandwidth', 'bottleneck_delay', 'access_bandwidth', 'access_delay']:
            args[arg_name] = value.strip() or ''
            command.extend([f'--{arg_name}', args[arg_name]])
        elif arg_name in ['load', 'save'] and value.strip():
            args[arg_name] = value.strip()
            command.extend([f'--{arg_name}', args[arg_name]])

    args['no_train'] = form.get('no_train') in ['true', 'on', '1', 'yes']
    args['debug'] = form.get('debug') in ['true', 'on', '1', 'yes']
    if args['no_train']:
        command.append('--no_train')
    if args['debug']:
        command.append('--debug')

    if 'action_mode' not in args:
        args['action_mode'] = 'default'
        command.extend(['--action_mode', 'default'])

    timeout_seconds = 600
    stdout_text = ''
    stderr_text = ''
    return_code = None
    timed_out = False

    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'

    try:
        result = subprocess.run(
            command,
            cwd=BASE_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        stdout_text = result.stdout
        stderr_text = result.stderr
        return_code = result.returncode
        
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout_text = exc.stdout or ''
        stderr_text = exc.stderr or ''
        return_code = None

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = get_run_dir(run_id)
    os.makedirs(run_dir, exist_ok=True)

    copy_plot(TRAIN_PLOT_PATH, os.path.join(run_dir, 'training.png'))
    copy_plot(TEST_PLOT_PATH, os.path.join(run_dir, 'test.png'))

    metadata = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'command': command,
        'args': args,
        'success': (return_code == 0 and not timed_out),
        'return_code': return_code,
        'timeout': timed_out,
        'stdout': stdout_text,
        'stderr': stderr_text,
        'plots': {
            'training': build_plot_urls(run_id)['training'],
            'test': build_plot_urls(run_id)['test'],
        },
    }
    record_history(metadata)

    response = {
        'success': metadata['success'],
        'return_code': return_code,
        'timeout': timed_out,
        'command': command,
        'stdout': stdout_text,
        'stderr': stderr_text,
        'training_plot_url': build_plot_urls()['training'],
        'test_plot_url': build_plot_urls()['test'],
        'history_run_id': run_id,
        'history_plot_urls': metadata['plots'],
    }

    return jsonify(response)


@app.route('/launch_netanim', methods=['POST'])
def launch_netanim():
    if not os.path.exists(NETANIM_BINARY):
        return jsonify({
            'success': False,
            'message': f'NetAnim executable not found: {NETANIM_BINARY}'
        }), 404

    if not os.path.exists(NETANIM_XML):
        return jsonify({
            'success': False,
            'message': f'Animation XML not found: {NETANIM_XML}'
        }), 404

    try:
        subprocess.Popen(
            [NETANIM_BINARY, NETANIM_XML],
            cwd=os.path.dirname(NETANIM_BINARY),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        return jsonify({
            'success': True,
            'message': 'NetAnim launched successfully.',
            'netanim_binary': NETANIM_BINARY,
            'animation_file': NETANIM_XML,
        })
    except OSError as exc:
        return jsonify({
            'success': False,
            'message': f'Failed to launch NetAnim: {exc}',
        }), 500


@app.route('/plot/<plot_type>')
def get_plot(plot_type):
    if plot_type == 'training':
        path = TRAIN_PLOT_PATH
    elif plot_type == 'test':
        path = TEST_PLOT_PATH
    else:
        return jsonify({'error': 'Unknown plot type'}), 404

    if not os.path.exists(path):
        return jsonify({'error': 'Plot not found'}), 404

    return send_file(path, mimetype='image/png')


@app.route('/history')
def get_history():
    return jsonify(load_history())


@app.route('/history/<run_id>/plot/<plot_type>')
def get_history_plot(run_id, plot_type):
    if plot_type == 'training':
        filename = 'training.png'
    elif plot_type == 'test':
        filename = 'test.png'
    else:
        return jsonify({'error': 'Unknown plot type'}), 404

    path = os.path.join(get_run_dir(run_id), filename)
    if not os.path.exists(path):
        return jsonify({'error': 'Plot not found'}), 404

    return send_file(path, mimetype='image/png')


@app.route('/status')
def status():
    return jsonify({
        'script_path': SCRIPT_PATH,
        'training_plot': TRAIN_PLOT_PATH,
        'test_plot': TEST_PLOT_PATH,
        'python_exec': PYTHON_EXECUTABLE,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

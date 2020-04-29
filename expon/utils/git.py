
import subprocess

def check_working_tree():
    try:
        subprocess.check_output(['git', 'diff', '--exit-code'])

    except subprocess.CalledProcessError as e:
        print('called process error')
        out_bytes = e.output       # Output generated before error
        code = e.returncode   # Return code
        raise Exception('Your working tree is not empty, please commit all changes.')
    else:
        return True

def get_git_revision_short_hash():
    try:
        commit_id = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except subprocess.CalledProcessError:
        raise Exception('Somthing is wrong with your git workspace!')
    else:
        return commit_id
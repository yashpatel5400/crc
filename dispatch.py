# VERY hacky script but hey, gets the job done
import libtmux
from uci_datasets import all_datasets

setups = ["airfoil", "load_pos", "pendulum", "battery", "fusion"]

server = libtmux.Server()

for setup in setups:
    server.new_session(attach=False)
    session = server.sessions[-1]
    p = session.attached_pane
    p.send_keys("conda activate spectral", enter=True)
    cmd = f"python gen_data.py --setup {setup} && python train.py --setup {setup} && python crc.py --setup {setup}"
    p.send_keys(cmd, enter=True)
    print(f"{cmd}")

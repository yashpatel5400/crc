# VERY hacky script but hey, gets the job done
import libtmux

approaches = ["box", "ptc_b", "ellipsoid", "ptc_e", "crc"]
server = libtmux.Server()
for approach in approaches:
    server.new_session(attach=False)
    session = server.sessions[-1]
    p = session.attached_pane
    p.send_keys("conda activate chig", enter=True)
    cmd = f"python crc.py --approach {approach}"
    p.send_keys(cmd, enter=True)
    print(f"{cmd}")
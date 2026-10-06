import argparse
import threading
import webbrowser

from src.web import create_app


def main():
    p = argparse.ArgumentParser(description="Hybrid Control Arm web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5050, help="5000 is used by the AirPlay Receiver on macOS")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = p.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    print(f"\nHybrid Control Arm is running. Open {url}\nStop it with Ctrl+C.\n", flush=True)
    if not args.no_browser:
        threading.Timer(1.2, webbrowser.open, args=(url,)).start()
    create_app().run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()

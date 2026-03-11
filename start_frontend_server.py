from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

HOST = "0.0.0.0"
PORT = 5500


if __name__ == "__main__":
    print("Frontend server started at http://localhost:5500")
    server = ThreadingHTTPServer((HOST, PORT), SimpleHTTPRequestHandler)
    server.serve_forever()

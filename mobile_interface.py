# mobile_interface.py - Mobile App Communication Module
import socket
import threading
import queue
import json
import struct
import time
import numpy as np
from typing import Optional, Callable, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MobileAudioProtocol:
    """Protocol for mobile audio streaming"""
    
    # Message types
    MSG_AUDIO_DATA = 0x01
    MSG_START_SESSION = 0x02
    MSG_END_SESSION = 0x03
    MSG_HEARTBEAT = 0x04
    MSG_ERROR = 0x05
    MSG_METADATA = 0x06
    
    # Audio format constants
    SAMPLE_RATE = 16000
    CHANNELS = 1
    SAMPLE_WIDTH = 2  # 16-bit audio
    
    @staticmethod
    def pack_message(msg_type: int, data: bytes) -> bytes:
        """Pack message with header: [msg_type:1][length:4][data:length]"""
        header = struct.pack('<BI', msg_type, len(data))
        return header + data
    
    @staticmethod
    def unpack_message(data: bytes) -> tuple[int, bytes]:
        """Unpack message header and return (msg_type, payload)"""
        if len(data) < 5:
            raise ValueError("Message too short")
        
        msg_type, length = struct.unpack('<BI', data[:5])
        
        if len(data) < 5 + length:
            raise ValueError("Incomplete message")
        
        payload = data[5:5+length]
        return msg_type, payload

class MobileClient:
    """Represents a connected mobile client"""
    
    def __init__(self, client_socket: socket.socket, address: tuple, client_id: str):
        self.socket = client_socket
        self.address = address
        self.client_id = client_id
        self.is_active = True
        self.last_heartbeat = time.time()
        self.session_metadata = {}
        
    def send_message(self, msg_type: int, data: bytes = b'') -> bool:
        """Send message to mobile client"""
        try:
            message = MobileAudioProtocol.pack_message(msg_type, data)
            self.socket.send(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {self.client_id}: {e}")
            self.is_active = False
            return False
    
    def close(self):
        """Close client connection"""
        self.is_active = False
        try:
            self.socket.close()
        except:
            pass

class MobileAudioServer:
    """Socket server for receiving audio from mobile apps"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8888):
        self.host = host
        self.port = port
        self.server_socket: Optional[socket.socket] = None
        self.is_running = False
        
        # Client management
        self.clients: Dict[str, MobileClient] = {}
        self.client_lock = threading.Lock()
        
        # Audio data queue
        self.audio_queue = queue.Queue()
        
        # Callbacks
        self.on_audio_data: Optional[Callable[[str, np.ndarray, Dict], None]] = None
        self.on_client_connected: Optional[Callable[[str], None]] = None
        self.on_client_disconnected: Optional[Callable[[str], None]] = None
        
        # Threading
        self.server_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start the mobile audio server"""
        if self.is_running:
            logger.warning("Server already running")
            return
        
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            self.is_running = True
            
            # Start server thread
            self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
            self.server_thread.start()
            
            # Start heartbeat monitoring
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
            self.heartbeat_thread.start()
            
            logger.info(f"Mobile audio server started on {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the mobile audio server"""
        self.is_running = False
        
        # Close all client connections
        with self.client_lock:
            for client in self.clients.values():
                client.close()
            self.clients.clear()
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            self.server_socket = None
        
        logger.info("Mobile audio server stopped")
    
    def _server_loop(self):
        """Main server loop for accepting connections"""
        while self.is_running:
            try:
                client_socket, address = self.server_socket.accept()
                logger.info(f"New connection from {address}")
                
                # Start client handler
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
                
            except Exception as e:
                if self.is_running:
                    logger.error(f"Error accepting connection: {e}")
                break
    
    def _handle_client(self, client_socket: socket.socket, address: tuple):
        """Handle individual client connection"""
        client_id = f"{address[0]}:{address[1]}:{int(time.time())}"
        client = MobileClient(client_socket, address, client_id)
        
        with self.client_lock:
            self.clients[client_id] = client
        
        # Notify connection
        if self.on_client_connected:
            self.on_client_connected(client_id)
        
        try:
            # Message buffer
            buffer = b''
            
            while self.is_running and client.is_active:
                # Receive data
                data = client_socket.recv(4096)
                if not data:
                    break
                
                buffer += data
                
                # Process complete messages
                while len(buffer) >= 5:  # Minimum header size
                    try:
                        # Peek at message length
                        msg_type, length = struct.unpack('<BI', buffer[:5])
                        total_length = 5 + length
                        
                        if len(buffer) < total_length:
                            # Wait for more data
                            break
                        
                        # Extract complete message
                        message_data = buffer[:total_length]
                        buffer = buffer[total_length:]
                        
                        # Process message
                        self._process_message(client, message_data)
                        
                    except Exception as e:
                        logger.error(f"Error processing message from {client_id}: {e}")
                        break
        
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        
        finally:
            # Clean up client
            with self.client_lock:
                if client_id in self.clients:
                    del self.clients[client_id]
            
            client.close()
            
            # Notify disconnection
            if self.on_client_disconnected:
                self.on_client_disconnected(client_id)
            
            logger.info(f"Client {client_id} disconnected")
    
    def _process_message(self, client: MobileClient, message_data: bytes):
        """Process incoming message from mobile client"""
        try:
            msg_type, payload = MobileAudioProtocol.unpack_message(message_data)
            
            if msg_type == MobileAudioProtocol.MSG_AUDIO_DATA:
                self._handle_audio_data(client, payload)
            
            elif msg_type == MobileAudioProtocol.MSG_START_SESSION:
                self._handle_start_session(client, payload)
            
            elif msg_type == MobileAudioProtocol.MSG_END_SESSION:
                self._handle_end_session(client, payload)
            
            elif msg_type == MobileAudioProtocol.MSG_HEARTBEAT:
                self._handle_heartbeat(client, payload)
            
            elif msg_type == MobileAudioProtocol.MSG_METADATA:
                self._handle_metadata(client, payload)
            
            else:
                logger.warning(f"Unknown message type {msg_type} from {client.client_id}")
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Send error response
            error_msg = json.dumps({"error": str(e)}).encode('utf-8')
            client.send_message(MobileAudioProtocol.MSG_ERROR, error_msg)
    
    def _handle_audio_data(self, client: MobileClient, payload: bytes):
        """Handle incoming audio data"""
        try:
            # Convert bytes to numpy array (16-bit PCM)
            audio_data = np.frombuffer(payload, dtype=np.int16).astype(np.float32)
            
            # Normalize to [-1, 1]
            audio_data = audio_data / 32768.0
            
            # Add to processing queue
            self.audio_queue.put({
                'client_id': client.client_id,
                'audio_data': audio_data,
                'timestamp': time.time(),
                'metadata': client.session_metadata.copy()
            })
            
            # Notify callback
            if self.on_audio_data:
                self.on_audio_data(client.client_id, audio_data, client.session_metadata)
            
        except Exception as e:
            logger.error(f"Error handling audio data: {e}")
    
    def _handle_start_session(self, client: MobileClient, payload: bytes):
        """Handle session start message"""
        try:
            if payload:
                session_data = json.loads(payload.decode('utf-8'))
                client.session_metadata.update(session_data)
            
            logger.info(f"Session started for {client.client_id}")
            
            # Send acknowledgment
            ack = json.dumps({"status": "session_started"}).encode('utf-8')
            client.send_message(MobileAudioProtocol.MSG_START_SESSION, ack)
            
        except Exception as e:
            logger.error(f"Error handling start session: {e}")
    
    def _handle_end_session(self, client: MobileClient, payload: bytes):
        """Handle session end message"""
        try:
            logger.info(f"Session ended for {client.client_id}")
            
            # Send acknowledgment
            ack = json.dumps({"status": "session_ended"}).encode('utf-8')
            client.send_message(MobileAudioProtocol.MSG_END_SESSION, ack)
            
        except Exception as e:
            logger.error(f"Error handling end session: {e}")
    
    def _handle_heartbeat(self, client: MobileClient, payload: bytes):
        """Handle heartbeat message"""
        client.last_heartbeat = time.time()
        
        # Send heartbeat response
        response = json.dumps({"timestamp": time.time()}).encode('utf-8')
        client.send_message(MobileAudioProtocol.MSG_HEARTBEAT, response)
    
    def _handle_metadata(self, client: MobileClient, payload: bytes):
        """Handle metadata update"""
        try:
            if payload:
                metadata = json.loads(payload.decode('utf-8'))
                client.session_metadata.update(metadata)
                logger.info(f"Metadata updated for {client.client_id}: {metadata}")
            
        except Exception as e:
            logger.error(f"Error handling metadata: {e}")
    
    def _heartbeat_monitor(self):
        """Monitor client heartbeats and remove stale connections"""
        while self.is_running:
            try:
                current_time = time.time()
                stale_clients = []
                
                with self.client_lock:
                    for client_id, client in self.clients.items():
                        if current_time - client.last_heartbeat > 30:  # 30 second timeout
                            stale_clients.append(client_id)
                
                # Remove stale clients
                for client_id in stale_clients:
                    logger.warning(f"Removing stale client: {client_id}")
                    with self.client_lock:
                        if client_id in self.clients:
                            self.clients[client_id].close()
                            del self.clients[client_id]
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
    
    def get_audio_data(self, timeout: float = None) -> Optional[Dict]:
        """Get audio data from queue (blocking)"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_connected_clients(self) -> Dict[str, Dict]:
        """Get information about connected clients"""
        with self.client_lock:
            return {
                client_id: {
                    'address': client.address,
                    'last_heartbeat': client.last_heartbeat,
                    'session_metadata': client.session_metadata.copy()
                }
                for client_id, client in self.clients.items()
            }
    
    def send_to_client(self, client_id: str, msg_type: int, data: bytes = b'') -> bool:
        """Send message to specific client"""
        with self.client_lock:
            if client_id in self.clients:
                return self.clients[client_id].send_message(msg_type, data)
        return False
    
    def broadcast_message(self, msg_type: int, data: bytes = b''):
        """Broadcast message to all connected clients"""
        with self.client_lock:
            for client in self.clients.values():
                client.send_message(msg_type, data)

class MobileAudioStream:
    """Mobile-compatible audio stream interface"""
    
    def __init__(self, mobile_server: MobileAudioServer):
        self.mobile_server = mobile_server
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # Set up mobile server callbacks
        self.mobile_server.on_audio_data = self._on_mobile_audio
        
    def start_recording(self):
        """Start receiving audio from mobile clients"""
        self.is_recording = True
        logger.info("Started mobile audio recording")
    
    def stop_recording(self):
        """Stop receiving audio from mobile clients"""
        self.is_recording = False
        logger.info("Stopped mobile audio recording")
    
    def _on_mobile_audio(self, client_id: str, audio_data: np.ndarray, metadata: Dict):
        """Handle audio data from mobile client"""
        if self.is_recording:
            self.audio_queue.put({
                'audio_data': audio_data,
                'client_id': client_id,
                'timestamp': time.time(),
                'metadata': metadata
            })
    
    def get_audio_chunk(self, timeout: float = None) -> Optional[np.ndarray]:
        """Get audio chunk for processing"""
        try:
            audio_item = self.audio_queue.get(timeout=timeout)
            return audio_item['audio_data']
        except queue.Empty:
            return None
    
    def get_audio_with_metadata(self, timeout: float = None) -> Optional[Dict]:
        """Get audio chunk with metadata"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

# Example usage
if __name__ == "__main__":
    def on_audio_received(client_id: str, audio_data: np.ndarray, metadata: Dict):
        print(f"Received audio from {client_id}: {len(audio_data)} samples, metadata: {metadata}")
    
    def on_client_connected(client_id: str):
        print(f"Client connected: {client_id}")
    
    def on_client_disconnected(client_id: str):
        print(f"Client disconnected: {client_id}")
    
    # Create and start mobile audio server
    server = MobileAudioServer(host='0.0.0.0', port=8888)
    server.on_audio_data = on_audio_received
    server.on_client_connected = on_client_connected
    server.on_client_disconnected = on_client_disconnected
    
    try:
        server.start()
        print("Mobile audio server running... Press Ctrl+C to stop")
        
        # Keep server running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()
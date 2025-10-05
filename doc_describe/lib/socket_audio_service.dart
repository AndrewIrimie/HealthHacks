import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

/// Service for managing TCP socket connection to the Python audio server
class SocketAudioService {
  Socket? _socket;
  bool _isConnected = false;
  StreamSubscription? _socketSubscription;
  
  // Connection settings
  String _host = 'localhost';
  int _port = 8080;
  
  // Message types - must match Python server
  static const int MSG_AUDIO_DATA = 0x01;
  static const int MSG_START_SESSION = 0x02;
  static const int MSG_END_SESSION = 0x03;
  static const int MSG_HEARTBEAT = 0x04;
  static const int MSG_ERROR = 0x05;
  static const int MSG_METADATA = 0x06;
  
  // Status callbacks
  Function(String)? onStatusUpdate;
  Function(String)? onError;
  Function()? onConnected;
  Function()? onDisconnected;
  
  // Connection state
  bool get isConnected => _isConnected;
  String get connectionInfo => '$_host:$_port';
  
  /// Update server connection settings
  void updateServerSettings(String host, int port) {
    _host = host;
    _port = port;
  }
  
  /// Connect to the Python audio server
  Future<bool> connect() async {
    try {
      onStatusUpdate?.call('Connecting to $_host:$_port...');
      
      _socket = await Socket.connect(_host, _port);
      _isConnected = true;
      
      // Listen for server responses
      _socketSubscription = _socket!.listen(
        _handleServerMessage,
        onError: _handleSocketError,
        onDone: _handleSocketDisconnection,
      );
      
      onStatusUpdate?.call('Connected to server');
      onConnected?.call();
      
      return true;
    } catch (e) {
      _isConnected = false;
      onError?.call('Failed to connect: $e');
      return false;
    }
  }
  
  /// Disconnect from the server
  Future<void> disconnect() async {
    if (_isConnected) {
      sendEndSession();
      await Future.delayed(const Duration(milliseconds: 100)); // Give time for message to send
    }
    
    await _socketSubscription?.cancel();
    await _socket?.close();
    _socket = null;
    _isConnected = false;
    onDisconnected?.call();
    onStatusUpdate?.call('Disconnected');
  }
  
  /// Pack message with binary protocol: [msg_type:1][length:4][data:length]
  Uint8List _packMessage(int msgType, Uint8List data) {
    final length = data.length;
    final message = Uint8List(5 + length);
    
    // Message type (1 byte)
    message[0] = msgType;
    
    // Length (4 bytes, little endian)
    final lengthBytes = Uint8List(4);
    lengthBytes.buffer.asByteData().setUint32(0, length, Endian.little);
    message.setRange(1, 5, lengthBytes);
    
    // Data
    message.setRange(5, 5 + length, data);
    
    return message;
  }
  
  /// Send a message to the server
  bool _sendMessage(int msgType, Uint8List data) {
    if (!_isConnected || _socket == null) {
      onError?.call('Cannot send message: not connected');
      return false;
    }
    
    try {
      final message = _packMessage(msgType, data);
      _socket!.add(message);
      return true;
    } catch (e) {
      onError?.call('Failed to send message: $e');
      return false;
    }
  }
  
  /// Send session start message
  Future<bool> sendStartSession({Map<String, dynamic>? metadata}) async {
    final sessionData = metadata ?? {
      'client_type': 'flutter',
      'version': '1.0',
      'platform': Platform.operatingSystem,
      'timestamp': DateTime.now().toIso8601String(),
    };
    
    final jsonData = utf8.encode(jsonEncode(sessionData));
    final success = _sendMessage(MSG_START_SESSION, Uint8List.fromList(jsonData));
    
    if (success) {
      onStatusUpdate?.call('Session started');
    }
    
    return success;
  }
  
  /// Send raw audio data
  bool sendAudioData(Uint8List audioBytes) {
    return _sendMessage(MSG_AUDIO_DATA, audioBytes);
  }
  
  /// Send session end message
  bool sendEndSession() {
    final success = _sendMessage(MSG_END_SESSION, Uint8List(0));
    if (success) {
      onStatusUpdate?.call('Session ended');
    }
    return success;
  }
  
  /// Send heartbeat to keep connection alive
  bool sendHeartbeat() {
    return _sendMessage(MSG_HEARTBEAT, Uint8List(0));
  }
  
  /// Send metadata message
  bool sendMetadata(Map<String, dynamic> metadata) {
    final jsonData = utf8.encode(jsonEncode(metadata));
    return _sendMessage(MSG_METADATA, Uint8List.fromList(jsonData));
  }
  
  /// Handle incoming messages from server
  void _handleServerMessage(Uint8List data) {
    // For now, just log that we received data
    // In a full implementation, you'd parse server responses
    onStatusUpdate?.call('Received ${data.length} bytes from server');
  }
  
  /// Handle socket errors
  void _handleSocketError(error) {
    _isConnected = false;
    onError?.call('Socket error: $error');
  }
  
  /// Handle socket disconnection
  void _handleSocketDisconnection() {
    _isConnected = false;
    onDisconnected?.call();
    onStatusUpdate?.call('Connection lost');
  }
  
  /// Start periodic heartbeat
  Timer? _heartbeatTimer;
  
  void startHeartbeat({Duration interval = const Duration(seconds: 30)}) {
    _heartbeatTimer?.cancel();
    _heartbeatTimer = Timer.periodic(interval, (_) {
      if (_isConnected) {
        sendHeartbeat();
      }
    });
  }
  
  void stopHeartbeat() {
    _heartbeatTimer?.cancel();
    _heartbeatTimer = null;
  }
  
  /// Dispose of resources
  void dispose() {
    stopHeartbeat();
    disconnect();
  }
}

/// Helper class for connection status
class ConnectionStatus {
  static const String idle = 'Idle';
  static const String connecting = 'Connecting...';
  static const String connected = 'Connected';
  static const String recording = 'Recording & Streaming';
  static const String disconnecting = 'Disconnecting...';
  static const String error = 'Connection Error';
}
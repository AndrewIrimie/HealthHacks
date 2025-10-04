#!/usr/bin/env python3
# test_mobile_connection.py - Test script for mobile socket server
import socket
import struct
import time
import threading
from mobile_interface import MobileAudioProtocol

def test_mobile_connection():
    """Test connection to mobile socket server"""
    
    # Connection settings
    HOST = 'localhost'
    PORT = 8080
    
    print(f"Testing connection to mobile server at {HOST}:{PORT}")
    
    try:
        # Create socket connection
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(10)  # 10 second timeout
        
        print("Connecting to server...")
        client_socket.connect((HOST, PORT))
        print("✅ Connected successfully!")
        
        # Send start session message
        print("Sending start session message...")
        start_msg = MobileAudioProtocol.pack_message(
            MobileAudioProtocol.MSG_START_SESSION, 
            b'{"client_type": "test", "version": "1.0"}'
        )
        client_socket.send(start_msg)
        print("✅ Start session message sent")
        
        # Send heartbeat
        print("Sending heartbeat...")
        heartbeat_msg = MobileAudioProtocol.pack_message(
            MobileAudioProtocol.MSG_HEARTBEAT, 
            b''
        )
        client_socket.send(heartbeat_msg)
        print("✅ Heartbeat sent")
        
        # Send dummy audio data
        print("Sending dummy audio data...")
        dummy_audio = b'\x00' * 1024  # 1024 bytes of silence
        audio_msg = MobileAudioProtocol.pack_message(
            MobileAudioProtocol.MSG_AUDIO_DATA,
            dummy_audio
        )
        client_socket.send(audio_msg)
        print("✅ Audio data sent")
        
        # Send end session
        print("Sending end session message...")
        end_msg = MobileAudioProtocol.pack_message(
            MobileAudioProtocol.MSG_END_SESSION,
            b''
        )
        client_socket.send(end_msg)
        print("✅ End session message sent")
        
        # Give server time to process
        time.sleep(2)
        
        # Close connection
        client_socket.close()
        print("✅ Connection closed successfully")
        
        return True
        
    except ConnectionRefusedError:
        print("❌ Connection refused - is the server running?")
        return False
    except socket.timeout:
        print("❌ Connection timeout")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_protocol_messages():
    """Test protocol message packing/unpacking"""
    
    print("\nTesting protocol message packing/unpacking...")
    
    # Test different message types
    test_cases = [
        (MobileAudioProtocol.MSG_START_SESSION, b'{"test": "data"}'),
        (MobileAudioProtocol.MSG_AUDIO_DATA, b'\x00\x01\x02\x03'),
        (MobileAudioProtocol.MSG_HEARTBEAT, b''),
        (MobileAudioProtocol.MSG_END_SESSION, b''),
        (MobileAudioProtocol.MSG_METADATA, b'{"sample_rate": 16000}')
    ]
    
    for msg_type, data in test_cases:
        try:
            # Pack message
            packed = MobileAudioProtocol.pack_message(msg_type, data)
            
            # Unpack message
            unpacked_type, unpacked_data = MobileAudioProtocol.unpack_message(packed)
            
            # Verify
            assert unpacked_type == msg_type, f"Type mismatch: {unpacked_type} != {msg_type}"
            assert unpacked_data == data, f"Data mismatch: {unpacked_data} != {data}"
            
            print(f"✅ Message type {msg_type} - OK")
            
        except Exception as e:
            print(f"❌ Message type {msg_type} - Error: {e}")
            return False
    
    print("✅ All protocol tests passed")
    return True

def main():
    """Main test function"""
    
    print("🔌 Mobile Socket Server Connection Test")
    print("=" * 50)
    
    # Test protocol first
    if not test_protocol_messages():
        print("\n❌ Protocol tests failed")
        return 1
    
    # Test actual connection
    print("\n" + "=" * 50)
    if not test_mobile_connection():
        print("\n❌ Connection test failed")
        print("\nTo start the server, run:")
        print("python main.py --mobile-only")
        return 1
    
    print("\n✅ All tests passed!")
    print("\nYour mobile app can now connect to:")
    print(f"📱 Host: localhost")
    print(f"📱 Port: 8080")
    print(f"📱 Protocol: TCP Socket")
    
    print("\nMessage format:")
    print("📱 [msg_type:1 byte][length:4 bytes][data:length bytes]")
    
    print("\nMessage types:")
    print(f"📱 START_SESSION: {MobileAudioProtocol.MSG_START_SESSION}")
    print(f"📱 AUDIO_DATA: {MobileAudioProtocol.MSG_AUDIO_DATA}")
    print(f"📱 END_SESSION: {MobileAudioProtocol.MSG_END_SESSION}")
    print(f"📱 HEARTBEAT: {MobileAudioProtocol.MSG_HEARTBEAT}")
    print(f"📱 METADATA: {MobileAudioProtocol.MSG_METADATA}")
    
    return 0

if __name__ == "__main__":
    exit(main())
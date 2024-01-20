#


## TOTP
```python 
TOTP()
```


---
TOTP class that inherits from the OTP class. It provides the functionality for generating and verifying Time-based One-Time Passwords (TOTP).

Methods:
get_qr: Generates a QR code for the TOTP.
verify: Verifies the TOTP.


**Methods:**


### .get_qr
```python
.get_qr(
   secret, name, issuer
)
```

---
Generates a QR code for the TOTP.

This method uses the pyotp library to generate a provisioning URI for the TOTP. This URI can be converted into a QR code.


**Args**

* **secret** (str) : The secret key for the TOTP.
* **name** (str) : The name of the TOTP.
* **issuer** (str) : The issuer of the TOTP.


**Returns**

* **str**  : The provisioning URI for the TOTP.


### .verify
```python
.verify(
   secret, otp
)
```

---
Verifies the TOTP.

This method uses the pyotp library to verify the TOTP.


**Args**

* **secret** (str) : The secret key for the TOTP.
* **otp** (str) : The TOTP to be verified.


**Returns**

* **bool**  : True if the TOTP is valid, False otherwise.


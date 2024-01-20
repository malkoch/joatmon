#


## OTP
```python 
OTP()
```


---
OTP class that inherits from the Plugin class. It provides the functionality for generating and verifying OTPs.

Methods:
get_qr: Generates a QR code for the OTP.
verify: Verifies the OTP.


**Methods:**


### .get_qr
```python
.get_qr(
   secret, name, issuer
)
```

---
Generates a QR code for the OTP.

This method should be implemented in the child classes.


**Args**

* **secret** (str) : The secret key for the OTP.
* **name** (str) : The name of the OTP.
* **issuer** (str) : The issuer of the OTP.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .verify
```python
.verify(
   secret, otp
)
```

---
Verifies the OTP.

This method should be implemented in the child classes.


**Args**

* **secret** (str) : The secret key for the OTP.
* **otp** (str) : The OTP to be verified.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


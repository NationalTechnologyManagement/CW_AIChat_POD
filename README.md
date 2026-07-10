# Hercules Internal

Technician-facing Hercules pod for ConnectWise Manage.

## ScreenConnect setup

Install the **RESTful API Manager** extension in ScreenConnect, then set:

- `RESTfulAuthenticationSecret`: a new, long random string
- `RESTfulUserName`: `Hercules`
- `RESTfulAllowedOrigin`: the public origin of this Hercules service, such as
  `https://your-hercules-service.up.railway.app`

Add matching values to the Hercules service environment:

```env
SCREENCONNECT_BASE_URL=https://your-instance.screenconnect.com
SCREENCONNECT_API_SECRET=the-value-from-RESTfulAuthenticationSecret
SCREENCONNECT_ORIGIN=https://your-hercules-service.up.railway.app
```

The pod resolves each ConnectWise configuration attached to the current ticket
by computer name. **ScreenConnect** launches normal control. **Backstage** opens
ScreenConnect's Join with Options dialog, where the technician can select the
Backstage logon session.

The Manage API member needs inquire access to Companies > Configurations and
Service Desk > Service Tickets. ScreenConnect technicians still authenticate
normally and must have permission to view and join the matched Access session.

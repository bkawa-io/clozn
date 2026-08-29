# CLOZN Studio v3

v3 is a separate Sessions application. Its build output is `studio/v3`; the existing Studio build and
default serving path are unchanged.

```sh
pnpm install
pnpm check
pnpm build
```

The app reads persisted session identity from `GET /sessions` and session detail from
`GET /sessions/<id>`. Conversation turns come from the paginated `GET /sessions/<id>/trace` projection;
the client fails closed when any page does not match its contract.

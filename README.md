# ModelWarehouse

---

### Table of Contents
  - [Introduction] (#introduction)
  - [Features] (#features)
  - [Usage] (#usage)
  - [Future] (#future)

## Introduction

Modelwarehouse is an ML operational tool that provides a native object database for specialized read/write capability with defined application to ML models.

## Features
- Model objects are byte serialized and stored as persistent, python objects.
- Database built on concept of 'Model' and 'Project' objects.
- API allows for detailed search based on metadata associated with stored objects.
- Flexible backend.  Local filestorage or relational database (postgres, sqlite, mysql, oracle)
  - Automated relational database setup - schemas, keys, tables
- Read/Write functionality is entirely pythonic/code oriented (dict like interface), no SQL.
- Transactions are *atomic* and *isolated* - applicationf scales across threads, processes, and users/machines while preventing conflicting changes

## Usage


### Database Setup

If using filestorage requires '/path/to/<mystoragefile>.fs'.

If using relational database requires '/path/to/<mydatabaseconfig>.toml' with relevant config formatted as below:

#### POSTGRES

``` xml
<relstorage>
  <postgresql>
    # The dsn is optional, as are each of the parameters in the dsn.
    dsn dbname='zodb' user='username' host='localhost' password='pass'
  </postgresql>
</relstorage>
```

#### SQLITE

``` xml
<relstorage>
    keep-history false
    cache-local-mb 0
    <sqlite3>
       data-dir /path/to/database/
    </sqlite3>
</relstorage>
```

#### MYSQL

``` xml
<relstorage>
  <mysql>
    # Most of the options provided by MySQLdb are available.
    # See component.xml.
    db zodb
  </mysql>
</relstorage>
```

#### ORACLE

``` xml
 <relstorage>
   <oracle>
     user username
     password pass
     dsn XE
   </oracle>
</relstorage>
```

### Model/Project Creation

### Handler Setup

Database handler setup.  All operations/setup are handled through this object.


#### FileStorage

``` python
from modelwarehouse.controller import Depot

log_filename = "my_log_name.log" # or None
log_filepath = "/path/to/log" # or None
path_to_configuration = "/path/to/<my_filestorage>.fs"

my_depot = Depot(path_to_configuration, log_filename, log_filepath)

```

#### Relational Database

``` python

from modelwarehouse.controller import Depot

log_filename = "my_log_name.log" # or None
log_filepath = "/path/to/log" # or None
path_to_configuration = "/path/to/<my_database_config>.toml"

my_depot = Depot(path_to_configuration, log_filename, log_filepath)

```


### Handler Operations

## Future
- Complete documentation
- Complete packaging requirements -  pyproject.toml, wheel generation etc
- Expand available 'meta_data' objects.
  - Add ML model object inference in populating 'meta_data' fields
- Optimize search operations through use of lightweight seconday BTrees

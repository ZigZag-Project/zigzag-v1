# Private ZigZag master
Some quick notes to owners to keep the code tidy and clean

## Developers forks
As a developer, fork privately this repository on your own account.


In order to pull updates from this repository, you have to add it as a remote upstream. In order to do so:

```git remote add upstream https://github.com/ZigZag-Project/zigzag_private_master```

Whenever updates are ready, pull them from upstream

```git pull origin [branch name]```

Since we will have different branches for each ongoing project/developer you can pull the code from a specific branch (``[branch name]``) or from the master itself (``master`` instead of ``[branch name]``)

### Pull requests from developer forks

Whenever an update is ready (__debugged, cleaned and commented__) it can be pushed to the private master repository. From the developer fork this can be done with a *pull request*. It is good practice to make a pull request to a *separate branch than the master* (created only for the pull request purpose) and then merge the updates with the master.

If you want to make a pull request for a specific branch of the private repository (be it a company one or any other one) you have to be careful to specify it when the pull request is being made (there is an option to do it very easily if the pull request is made with the browser and not via terminal)

## Company forks
If you are responsible for a company, you have to fork this repository privately on your own account and give it a name related to the company.

__Any user added as *outside collaborator* to the fork will not be able to pull from the private master repository__. It will be the duty of the responsible to pull updates from the private master.

Similarly to the developer fork, in order to pull updates from this repository, you have to add it as a remote upstream. In order to do so:

```git remote add upstream https://github.com/ZigZag-Project/zigzag_private_master```

Whenever updates are ready, pull them from upstream

```git pull origin [branch name]```

Since we will have different branches for each ongoing project/developer you can pull the code from a specific branch (``[branch name]``) or from the master itself (``master`` instead of ``[branch name]``).

### Outside collaborators

As an owner, you will be able to add external users as *outside collaborators*. These users will have access to the fork you are responsible for: they will be able to push/pull and see all the branches of the company fork (unless set differently). They will however not be able to pull from the private master.
